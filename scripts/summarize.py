"""Summarize Pangram scores by year and compare years (exploratory pilot).

Usage:  python -I summarize.py results/A.scores.jsonl results/B.scores.jsonl ...
Writes: results/pilot_summary.md (atomic) and prints the same.

Outcomes (per abstract):
  label_ai      prediction_short == "AI"
  label_any     prediction_short in {"AI", "Mixed"}   (any AI involvement flagged)
  frac_ai       fraction_ai (share of text classified AI-generated)
  frac_any      fraction_ai + fraction_ai_assisted
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(paths):
    by_year = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("stage") != "STAGE_SUCCESS":
                continue
            r["label_ai"] = r["prediction_short"] == "AI"
            r["label_any"] = r["prediction_short"] in ("AI", "Mixed")
            r["frac_any"] = (r["fraction_ai"] or 0) + (r["fraction_ai_assisted"] or 0)
            by_year[r["year"]].append(r)
    return dict(sorted(by_year.items()))


def pct_ci(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{100 * k / n:.1f}% ({100 * lo:.1f}–{100 * hi:.1f})"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    base_years = (2020, 2021)  # pooled pre-ChatGPT baseline, as pre-registered
    if args and args[0].startswith("--baseline="):
        base_years = tuple(int(x) for x in args.pop(0).split("=", 1)[1].split(","))
    data = load(args)
    years = list(data)
    out = ["# Pangram pilot summary (exploratory, not pre-registered)", "",
           "Model `pangram-4.1`. Proportions with Wilson 95% CI.", "",
           "| Year | n | Label AI | Label AI or Mixed | ≥1 AI or AI-assisted segment | Mean fraction AI | Mean fraction AI+assisted |",
           "|---|---|---|---|---|---|---|"]
    for y in years:
        rs = data[y]; n = len(rs)
        out.append(f"| {y} | {n} | {pct_ci(sum(r['label_ai'] for r in rs), n)} | "
                   f"{pct_ci(sum(r['label_any'] for r in rs), n)} | "
                   # segment counts, NOT fraction_ai > 0 (fractions carry float dust ~1e-45 on human text)
                   f"{pct_ci(sum((r['num_ai_segments'] or 0) + (r['num_ai_assisted_segments'] or 0) > 0 for r in rs), n)} | "
                   f"{np.mean([r['fraction_ai'] for r in rs]):.3f} | {np.mean([r['frac_any'] for r in rs]):.3f} |")
    out += ["", "Headline labels: " + "; ".join(
        f"{y}: " + ", ".join(f"{k} {v}" for k, v in Counter(r['headline'] for r in data[y]).most_common())
        for y in years)]

    if len(years) >= 2:
        byrs = [y for y in years if y in base_years]
        base = "+".join(str(y) for y in byrs)
        post = [y for y in years if y not in base_years]
        A = [r for y in byrs for r in data[y]]
        tests = []
        binary = (("label_ai", "Label AI"), ("label_any", "Label AI or Mixed"))
        for key, name in binary:                  # each later year vs baseline
            ka = sum(r[key] for r in A)
            for y in post:
                B = data[y]; kb = sum(r[key] for r in B)
                orr, p = fisher_exact([[kb, len(B) - kb], [ka, len(A) - ka]])
                tests.append((name, f"{y} vs {base}", f"Fisher exact, OR " +
                              (f"{orr:.2f}" if np.isfinite(orr) else f"inf ({base} had 0)"), p))
        if len(post) >= 3:                        # trend across post-AI years only
            for key, name in binary:
                yv = np.array([y - post[0] for y in post for _ in data[y]], dtype=float)
                ev = np.array([float(r[key]) for y in post for r in data[y]])
                fit = sm.Logit(ev, sm.add_constant(yv)).fit(disp=0)
                tests.append((name, f"trend {post[0]}–{post[-1]}",
                              f"logistic, OR per year {np.exp(fit.params[1]):.2f} "
                              f"(95% CI {np.exp(fit.conf_int()[1][0]):.2f}–{np.exp(fit.conf_int()[1][1]):.2f})",
                              fit.pvalues[1]))
        adj = multipletests([t[3] for t in tests], method="fdr_bh")[1]
        out += ["", "## Year comparisons", "",
                f"| Outcome | Comparison | Test / effect | p (raw) | p (BH-adjusted, {len(tests)} tests) |",
                "|---|---|---|---|---|"]
        for (name, comp, test, p), q in zip(tests, adj):
            out.append(f"| {name} | {comp} | {test} | {p:.2g} | {q:.2g} |")

    # descriptive only (too small for subgroup tests): label AI-or-Mixed by first-author country group
    clean = {}
    for y in years:
        for line in open(os.path.join(ROOT, "data", "clean", f"abstracts_{y}.jsonl"), encoding="utf-8"):
            r = json.loads(line); clean[r["id"]] = r
    out += ["", "## Descriptive only: AI-or-Mixed label by first-author country (no tests at this n)", "",
            "| Year | United States | China | Other |", "|---|---|---|---|"]
    for y in years:
        cells = []
        for grp in ("United States", "China", "Other"):
            rs = [r for r in data[y] if (clean[r["id"]]["first_author_country"] == grp) or
                  (grp == "Other" and clean[r["id"]]["first_author_country"] not in ("United States", "China"))]
            k = sum(r["label_any"] for r in rs)
            cells.append(f"{k}/{len(rs)}" + (f" ({100 * k / len(rs):.0f}%)" if rs else ""))
        out.append(f"| {y} | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    dest = os.path.join(ROOT, "results", "pilot_summary.md")
    with open(dest + ".tmp", "w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
        f.write(text)
    os.replace(dest + ".tmp", dest)
    print(text)


if __name__ == "__main__":
    main()
