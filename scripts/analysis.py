"""Statistics, tables and figures of the ASH abstract AI-text audit, computed from the Pangram results.

Reads results/*.scores.jsonl and results/*.raw.jsonl (from `reproduce.py rehydrate` or `score`)
and manifests/validation_rewrite.json and validation_fullgen.json. Writes:

  results/tables/by_year.csv         Figure 1 numbers: Pangram labels per year, share flagged
                                     (AI-written or partly AI-written) with a Wilson 95% interval,
                                     two-sided Fisher exact test of each later year against pooled
                                     2020-2021, Benjamini-Hochberg q-values across the later years
  results/tables/terms.csv           words and phrases that pass both term tests (see build_terms)
  results/tables/term_tests.json     the listing rule and the size of each BH family
  results/tables/ai_rewrites.csv     Figure 2 numbers
  results/tables/ai_written_controls.csv   Figure 3 numbers
  results/figures/figure1_flagged_by_year.svg / .pdf / .png
  results/figures/figure2_ai_rewrites.svg / .pdf / .png
  results/figures/figure3_ai_written_controls.svg / .pdf / .png

The tables and SVG figures are byte-identical on every platform; reproduce.py compares them with
manifests/expected_outputs.json. The PDF and PNG copies are for convenience and are not compared.
Figures 2 and 3 are drawn from counts in manifests/validation_rewrite.json and
manifests/validation_fullgen.json: the AI rewrites and the AI-written controls are derived from
ASH's abstracts and are not published.

Usage:
  python -I analysis.py                    # from results/
  python -I analysis.py --selftest         # no data needed: rebuild the three figures and their
                                           # tables from the published counts and compare
  python -I analysis.py --write-expected   # maintainer: record output hashes in the manifest
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BASELINE_YEARS = (2020, 2021)     # pooled pre-ChatGPT baseline
FIRST_YEAR = 2020
FULL_N = 500                      # planned per-year sample
NO_TEXT = {2022: "No abstract text in Crossref"}
MODEL = "Pangram 4.1"

# Term tests (Eric, 2026-11-14): two two-sided Fisher tests per 1-3-word term, BH within each family
ERA_FROM = 2023                   # highlighted vs unhighlighted text from these years on
FDR_Q = 0.05
MIN_TESTED = 10                   # family membership: >= 10 abstracts (year) / occurrences (highlighted)
MIN_ABSTRACTS = 6                 # listing floor: in >= 6 different highlighted abstracts
MIN_RATIO = 5.0                   # listing floor: > 5 times as common in highlighted text
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
WORD_RE = re.compile(r"[a-z]{2,}(?:['\-][a-z]+)*")

OUTPUTS = ("results/pilot_summary.md",
           "results/tables/by_year.csv", "results/tables/terms.csv", "results/tables/term_tests.json",
           "results/tables/ai_rewrites.csv", "results/tables/ai_written_controls.csv",
           "results/figures/figure1_flagged_by_year.svg", "results/figures/figure2_ai_rewrites.svg",
           "results/figures/figure3_ai_written_controls.svg")
SELFTEST_OUTPUTS = ("results/tables/by_year.csv", "results/tables/ai_rewrites.csv",
                    "results/tables/ai_written_controls.csv",
                    "results/figures/figure1_flagged_by_year.svg", "results/figures/figure2_ai_rewrites.svg",
                    "results/figures/figure3_ai_written_controls.svg")
EXPECTED = os.path.join(ROOT, "manifests", "expected_outputs.json")


# ---------------------------------------------------------------- loading

def load_labels(root=ROOT):
    """year -> {abstract id: Pangram label}, and id -> year, for every successfully scored abstract."""
    by_year, year_of = {}, {}
    for p in sorted(glob.glob(os.path.join(root, "results", "*.scores.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("stage") == "STAGE_SUCCESS":
                by_year.setdefault(r["year"], {})[r["id"]] = r["prediction_short"]
                year_of[r["id"]] = r["year"]
    return by_year, year_of


def load_raw(year_of, root=ROOT):
    """id -> (year, Pangram response) for every successfully scored abstract."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "*.raw.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            res = x["response"]
            if res.get("stage") != "STAGE_SUCCESS" or x["id"] in out or x["id"] not in year_of:
                continue
            out[x["id"]] = (year_of[x["id"]], res)
    return out


def check_raw(raw, by_year, year_of):
    missing = set(year_of) - set(raw)
    if missing:
        raise SystemExit(f"{len(missing)} scored abstracts have no raw Pangram response")
    bad = [i for i, (_, res) in raw.items() if res["prediction_short"] != by_year[year_of[i]][i]]
    if bad:
        raise SystemExit(f"{len(bad)} abstracts disagree between scores and raw files")


def counts_of(by_year):
    """year -> {n, ai, mixed} from per-abstract labels."""
    out = {}
    for y, labs in by_year.items():
        v = list(labs.values())
        out[y] = {"n": len(v), "ai": v.count("AI"), "mixed": v.count("Mixed")}
    return out


def counts_from_manifest():
    """year -> {n, ai, mixed} from the published counts (manifests/results_summary.csv)."""
    with open(os.path.join(ROOT, "manifests", "results_summary.csv"), encoding="utf-8", newline="") as f:
        return {int(r["year"]): {"n": int(r["n"]), "ai": int(r["ai"]), "mixed": int(r["mixed"])}
                for r in csv.DictReader(f)}


# ---------------------------------------------------------------- statistics

def year_rows(counts, last_year=None, pending=None):
    """Per-year rows for Figure 1, the pooled baseline, and the number of tests.

    Each scored later year is compared with pooled 2020-2021 by a two-sided Fisher exact test
    on flagged (AI or Mixed) vs not; q = Benjamini-Hochberg across those years."""
    pending = pending or {}
    last = last_year or max(counts)
    rows = []
    for y in range(FIRST_YEAR, last + 1):
        if y in NO_TEXT:
            rows.append({"year": y, "status": "nodata", "note": NO_TEXT[y]}); continue
        if y in pending and y not in counts:
            rows.append({"year": y, "status": "pending", "note": pending[y]}); continue
        if y not in counts:
            continue
        c = counts[y]
        n, ai, mixed = c["n"], c["ai"], c["mixed"]
        lo, hi = proportion_confint(ai + mixed, n, method="wilson")
        rows.append({"year": y, "status": "scored", "n": n, "ai": ai, "mixed": mixed,
                     "flag": ai + mixed, "lo": lo, "hi": hi, "partial": n < FULL_N})
    brows = [r for r in rows if r["year"] in BASELINE_YEARS and r["status"] == "scored"]
    bn, bk = sum(r["n"] for r in brows), sum(r["flag"] for r in brows)
    blo, bhi = proportion_confint(bk, bn, method="wilson")
    base = {"label": "–".join(str(r["year"]) for r in brows), "n": bn, "flag": bk, "lo": blo, "hi": bhi}
    for r in brows:
        r["is_base"] = True
    comp = [r for r in rows if r["status"] == "scored" and r["year"] not in BASELINE_YEARS]
    ps = [fisher_exact([[r["flag"], r["n"] - r["flag"]], [bk, bn - bk]])[1] for r in comp]
    for r, p in zip(comp, ps):
        r["p"] = float(p)
    if ps:
        for r, q in zip(comp, multipletests(ps, method="fdr_bh")[1]):
            r["q"] = float(q)
    return rows, base, len(ps)


def grams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            if all(WORD_RE.fullmatch(t) for t in tokens[i:i + n])]


def build_terms(raw):
    """Words (n=1) and phrases (n=2,3) that are significantly more common in highlighted
    than unhighlighted text AND significantly more common in latest-year abstracts than in
    2020-21 abstracts (two Fisher tests, BH within each family). Year shares are of all
    sampled abstracts in a year that contain the term anywhere."""
    n_year = Counter(y for y, _ in raw.values())
    years = sorted(n_year)
    latest = max(years)
    n_base = sum(n_year[y] for y in BASELINE_YEARS)
    toks = {aid: TOKEN_RE.findall(res["text"].lower()) for aid, (_, res) in raw.items()}
    in_year = defaultdict(Counter)            # year -> term -> abstracts containing it
    hl, hu = Counter(), Counter()             # term -> occurrences (ERA_FROM+ text)
    nhl, nhu = Counter(), Counter()           # n -> total n-gram slots, highlighted / not
    hl_abs = defaultdict(set)
    for n in (1, 2, 3):
        for aid, (y, res) in raw.items():
            for g in set(grams(toks[aid], n)):
                in_year[y][g] += 1
            if y < ERA_FROM:
                continue
            for w in res.get("windows", []):
                gs = grams(TOKEN_RE.findall(res["text"][w["start_index"]:w["end_index"]].lower()), n)
                if w["label"] == "Human Written":
                    hu.update(gs); nhu[n] += len(gs)
                else:
                    hl.update(gs); nhl[n] += len(gs)
                    for g in gs:
                        hl_abs[g].add(aid)
    size = lambda g: g.count(" ") + 1
    base = lambda g: sum(in_year[y][g] for y in BASELINE_YEARS)

    # year test: latest vs pooled baseline, abstract-level
    fam_a = sorted(g for g in set(in_year[latest]).union(*(in_year[y] for y in BASELINE_YEARS))
                   if in_year[latest][g] + base(g) >= MIN_TESTED)
    p_a = [fisher_exact([[in_year[latest][g], n_year[latest] - in_year[latest][g]],
                         [base(g), n_base - base(g)]])[1] for g in fam_a]
    q_a = dict(zip(fam_a, multipletests(p_a, method="fdr_bh")[1]))
    # highlighted test: highlighted vs unhighlighted ERA_FROM+ text, occurrence-level
    fam_b = sorted(g for g in set(hl) | set(hu) if hl[g] + hu[g] >= MIN_TESTED)
    p_b = [fisher_exact([[hl[g], nhl[size(g)] - hl[g]], [hu[g], nhu[size(g)] - hu[g]]])[1] for g in fam_b]
    q_b = dict(zip(fam_b, multipletests(p_b, method="fdr_bh")[1]))

    found = {}
    for g in set(q_a) & set(q_b):
        rate_hl, rate_hu = hl[g] / nhl[size(g)], hu[g] / nhu[size(g)]
        rose = in_year[latest][g] / n_year[latest] > base(g) / n_base
        ratio = rate_hl / ((hu[g] + 0.5) / nhu[size(g)])
        if (q_a[g] < FDR_Q and q_b[g] < FDR_Q and rose and rate_hl > rate_hu
                and len(hl_abs[g]) >= MIN_ABSTRACTS and ratio > MIN_RATIO):
            found[g] = {"t": g, "n": size(g), "k": len(hl_abs[g]), "h": hl[g], "u": hu[g],
                        "r": round(ratio, 1),
                        "qa": float(q_a[g]), "qb": float(q_b[g]),
                        "b": round(100 * base(g) / n_base, 1),
                        "s": [round(100 * in_year[y][g] / n_year[y], 1) for y in years]}
    # a longer term found in nearly the same highlighted abstracts (>= 80%) as a listed shorter
    # term inside it adds nothing ("underscore the" next to "underscore"): list only the shorter
    for g in sorted(found, key=lambda t: -size(t)):
        if any(o != g and size(o) < size(g) and f" {o} " in f" {g} " and found[g]["k"] >= 0.8 * found[o]["k"]
               for o in found):
            del found[g]
    key = lambda v: (-v["r"], v["t"])
    return {"years": years, "latest": latest, "base_label": "–".join(map(str, BASELINE_YEARS)),
            "words": sorted((v for v in found.values() if v["n"] == 1), key=key),
            "phrases": sorted((v for v in found.values() if v["n"] > 1), key=key),
            "rule": {"q": FDR_Q, "min_tested": MIN_TESTED, "abstracts": MIN_ABSTRACTS, "ratio": MIN_RATIO,
                     "hl_share": round(100 * nhl[1] / (nhl[1] + nhu[1]), 1),
                     "era_from": ERA_FROM, "n_year_tests": len(fam_a), "n_hl_tests": len(fam_b)}}


def load_validation():
    return json.load(open(os.path.join(ROOT, "manifests", "validation_rewrite.json"), encoding="utf-8"))


def load_fullgen():
    return json.load(open(os.path.join(ROOT, "manifests", "validation_fullgen.json"), encoding="utf-8"))


SOURCE_LABEL = {"Human": "human-written", "Mixed": "partly AI-written", "AI": "AI-written"}


def fullgen_bars(v):
    """Figure 3's bars: all controls, then the controls split by the label the detector gave the
    abstract each one was based on. (key, {AI, Mixed, Human} counts)."""
    return [("all", v["labels"])] + [(k, v["by_source_label"][k]) for k in ("Human", "Mixed", "AI")]


# ---------------------------------------------------------------- tables

def g8(x):
    """8 significant digits: Fisher p-values differ in the 16th digit between operating systems.
    Values below 1e-12 are float dust (Wilson's lower bound at 0 of n) and are written as 0."""
    return "" if x is None else ("0" if abs(x) < 1e-12 else f"{x:.8g}")


def gp(x):
    """p- and q-values: 8 significant digits however small (a p-value is never written as 0)."""
    return "" if x is None else f"{x:.8g}"


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
        f.write(text)
    os.replace(path + ".tmp", path)


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def table_by_year(rows, base):
    header = ["year", "n_scored", "ai_written", "partly_ai_written", "human_written", "flagged",
              "pct_flagged", "ci95_low_pct", "ci95_high_pct", "fisher_p_vs_2020_2021", "bh_q", "note"]
    out = []
    for r in rows:
        if r["status"] != "scored":
            out.append([r["year"]] + [""] * 10 + [r["note"]])
            continue
        note = "baseline" if r.get("is_base") else ("partial sample" if r["partial"] else "")
        out.append([r["year"], r["n"], r["ai"], r["mixed"], r["n"] - r["flag"], r["flag"],
                    f"{100 * r['flag'] / r['n']:.1f}", g8(100 * r["lo"]), g8(100 * r["hi"]),
                    gp(r.get("p")), gp(r.get("q")), note])
        if r["year"] == BASELINE_YEARS[-1]:
            out.append([base["label"].replace("–", "-"), base["n"], "", "", base["n"] - base["flag"], base["flag"],
                        f"{100 * base['flag'] / base['n']:.1f}", g8(100 * base["lo"]), g8(100 * base["hi"]),
                        "", "", "pooled baseline (comparison group)"])
    return csv_text(header, out)


def table_terms(terms):
    years = terms["years"]
    header = (["type", "term", "highlighted_abstracts", "highlighted_uses", "unhighlighted_uses",
               "times_as_common_highlighted", "q_highlighted_test", "q_year_test",
               f"pct_abstracts_{terms['base_label'].replace('–', '_')}"]
              + [f"pct_abstracts_{y}" for y in years])
    out = []
    for kind in ("words", "phrases"):
        for t in terms[kind]:
            out.append([kind[:-1], t["t"], t["k"], t["h"], t["u"], t["r"], gp(t["qb"]), gp(t["qa"]), t["b"]]
                       + t["s"])
    return csv_text(header, out)


def term_tests_json(terms):
    d = {"years": terms["years"], "latest_year": terms["latest"], "baseline": terms["base_label"],
         "rule": terms["rule"], "listed": {"words": len(terms["words"]), "phrases": len(terms["phrases"])},
         "about": ("Year test: abstract-level two-sided Fisher exact test, latest year vs pooled baseline, "
                   "BH across n_year_tests terms. Highlighted test: occurrence-level two-sided Fisher exact "
                   "test, Pangram-highlighted vs unhighlighted text from era_from on, BH across n_hl_tests "
                   "terms. Family = terms with >= min_tested uses. Listed if q < q on both tests in the "
                   "'more' direction, in >= abstracts highlighted abstracts, and > ratio times as common "
                   "in highlighted text.")}
    return json.dumps(d, ensure_ascii=False, indent=2) + "\n"


def table_rewrites(v):
    header = ["version", "n_scored", "ai_written", "partly_ai_written", "human_written", "pct_flagged",
              "ci95_low_pct", "ci95_high_pct", "note"]
    O, L, n = v["originals_same_abstracts"], v["labels"], v["n_scored"]
    lo, hi = v["flagged_ci95"]
    return csv_text(header, [
        ["original (ASH 2020)", n, O["AI"], O["Mixed"], O["Human"], f"{100 * (O['AI'] + O['Mixed']) / n:.1f}",
         "", "", "the same abstracts as written"],
        [f"rewritten by {v['generator']['model_name']}", n, L["AI"], L["Mixed"], L["Human"],
         f"{100 * (L['AI'] + L['Mixed']) / n:.1f}", g8(100 * lo), g8(100 * hi),
         f"{v['n_requested']} requested, {v['n_refused']} declined by Anthropic's safety filter"]])


def table_fullgen(v):
    header = ["abstracts", "n_scored", "ai_written", "partly_ai_written", "human_written", "pct_flagged",
              "ci95_low_pct", "ci95_high_pct", "note"]
    S = v["sources_same_abstracts"]
    ns = S["AI"] + S["Mixed"] + S["Human"]
    out = [[f"source abstracts as written (ASH {v['source_year']})", ns, S["AI"], S["Mixed"], S["Human"],
            f"{100 * (S['AI'] + S['Mixed']) / ns:.1f}", "", "", "the abstracts the controls were based on"]]
    for key, c in fullgen_bars(v):
        n = c["AI"] + c["Mixed"] + c["Human"]
        flag = c["AI"] + c["Mixed"]
        lo, hi = proportion_confint(flag, n, method="wilson")
        if key == "all":
            name = f"all controls written by {v['generator']['model_name']}"
            note = (f"{v['n_candidates']} source abstracts tried, {v['n_refused']} declined by Anthropic's "
                    f"safety filter, the first {v['n_scored']} completed used")
        else:
            name = f"controls based on an abstract labeled {SOURCE_LABEL[key]}"
            note = ""
        out.append([name, n, c["AI"], c["Mixed"], c["Human"], f"{100 * flag / n:.1f}",
                    g8(100 * lo), g8(100 * hi), note])
    return csv_text(header, out)


# ---------------------------------------------------------------- figures

# Okabe-Ito colors (safe for color-blind readers)
C_AI, C_MIXED, C_HUMAN = "#0072B2", "#D55E00", "#BBBBBB"
INK, MUTE, GRID = "#1a1a1a", "#5f5f5f", "#e3e3e3"
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",      # ships with matplotlib: same glyphs on every OS
        "font.size": 9,
        "svg.fonttype": "path",
        "svg.hashsalt": "artificial-ash",  # fixed ids inside the SVG
        "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
        "text.color": INK, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def q_text(q):
    if q >= 0.001:
        return f"q = {q:.2f}" if q >= 0.01 else f"q = {q:.3f}"
    m, e = f"{q:.1e}".split("e")
    return f"q = {m}×10{str(int(e)).translate(SUP)}"


def save(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    svg = io.StringIO()  # matplotlib writes SVG in text mode (CRLF on Windows); write LF on every OS
    fig.savefig(svg, format="svg", metadata={"Date": None})
    write_text(stem + ".svg", svg.getvalue())
    fig.savefig(stem + ".pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem + ".png", dpi=200, metadata={"Software": None})


def figure1(rows, base, stem):
    plt = plt_setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(7.0, 4.9))
    ax = fig.add_axes([0.10, 0.30, 0.87, 0.55])   # explicit layout: nothing is cropped on save
    xs = list(range(len(rows)))
    top = max((r["hi"] for r in rows if r["status"] == "scored"), default=0.1)
    ymax = min(1.0, (int(top * 10) + 1) / 10 + 0.05)
    bw = 0.62
    for x, r in zip(xs, rows):
        if r["status"] != "scored":
            ax.text(x, 0.012, "no abstract\ntext in\nCrossref" if r["status"] == "nodata" else "pending",
                    ha="center", va="bottom", fontsize=7.5, color=MUTE, linespacing=1.2)
            continue
        fa, fm = r["ai"] / r["n"], r["mixed"] / r["n"]
        if fa:
            ax.bar(x, fa, bw, color=C_AI, linewidth=0, zorder=2)
        if fm:
            ax.bar(x, fm, bw, bottom=fa, color=C_MIXED, linewidth=0, zorder=2)
        # Wilson's lower bound at 0 of n comes out a hair above 0 in floating point
        ax.errorbar(x, fa + fm, yerr=[[max(0.0, fa + fm - r["lo"])], [max(0.0, r["hi"] - fa - fm)]], fmt="none",
                    ecolor=INK, elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)
        ax.text(x, r["hi"] + 0.012, f"{100 * (fa + fm):.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, ymax)
    ticks = [t / 10 for t in range(0, int(ymax * 10) + 1)]
    ax.set_yticks(ticks, [f"{round(100 * t)}%" for t in ticks])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(xs, [str(r["year"]) for r in rows], fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("Abstracts flagged (AI-written or partly AI-written)")
    tr = ax.get_xaxis_transform()
    for x, r in zip(xs, rows):
        if r["status"] != "scored":
            continue
        ax.text(x, -0.10, f"n = {r['n']}", transform=tr, ha="center", va="top", fontsize=8, color=MUTE)
        if "q" in r:
            ax.text(x, -0.17, q_text(r["q"]), transform=tr, ha="center", va="top", fontsize=8, color=MUTE)
    bx = [x for x, r in zip(xs, rows) if r.get("is_base")]
    if bx:
        y0 = -0.165
        ax.plot([bx[0] - 0.3, bx[-1] + 0.3], [y0, y0], transform=tr, color=MUTE, linewidth=0.8, clip_on=False)
        ax.text((bx[0] + bx[-1]) / 2, y0 - 0.025, f"baseline, pooled: {base['flag']}/{base['n']}",
                transform=tr, ha="center", va="top", fontsize=8, color=MUTE)
    ax.legend(handles=[Patch(color=C_AI, label="AI-written"),
                       Patch(color=C_MIXED, label="Partly AI-written"),
                       Line2D([], [], color=INK, linewidth=1.1, marker="_", markersize=7, label="95% interval")],
              loc="upper left", frameon=False, fontsize=8.5, handlelength=1.4)
    fig.text(0.10, 0.955, "ASH abstracts flagged by an AI-text detector, by meeting year",
             fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.10, 0.905, f"Detector: {MODEL}. Random sample of abstracts from each annual meeting.",
             fontsize=8.5, color=MUTE, ha="left", va="top")
    fig.text(0.10, 0.025, "Bars: share labeled AI-written or partly AI-written, with Wilson 95% intervals.\n"
             "q: two-sided Fisher exact test of each year against pooled 2020–2021,\n"
             "Benjamini–Hochberg adjusted across the later years.",
             fontsize=7.5, color=MUTE, ha="left", va="bottom", linespacing=1.4)
    save(fig, stem)
    plt.close(fig)


def figure2(v, stem):
    plt = plt_setup()
    from matplotlib.patches import Patch
    O, L, n = v["originals_same_abstracts"], v["labels"], v["n_scored"]
    name = v["generator"]["model_name"]
    bars = [(f"As written for ASH 2020", O), (f"Rewritten by {name}", L)]
    fig = plt.figure(figsize=(7.0, 3.0))
    ax = fig.add_axes([0.32, 0.30, 0.63, 0.42])
    for i, (lab, c) in enumerate(bars):
        y = len(bars) - 1 - i
        left = 0.0
        for key, col, dark in (("AI", C_AI, True), ("Mixed", C_MIXED, True), ("Human", C_HUMAN, False)):
            w = c[key] / n
            if not w:
                continue
            ax.barh(y, w, 0.58, left=left, color=col, linewidth=0, zorder=2)
            if w >= 0.06:
                ax.text(left + w / 2, y, f"{100 * w:.1f}%", ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if dark else INK, zorder=3)
            left += w
    ax.set_yticks(range(len(bars)), [f"{lab}\n({n} abstracts)" for lab, _ in reversed(bars)], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(bars) - 0.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=C_AI, label="AI-written"), Patch(color=C_MIXED, label="Partly AI-written"),
                       Patch(color=C_HUMAN, label="Human-written")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=8.5, handlelength=1.4)
    fig.text(0.03, 0.955, "How the detector labels abstracts rewritten by AI",
             fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.875, f"Detector: {MODEL}.", fontsize=8.5, color=MUTE, ha="left", va="top")
    fig.text(0.03, 0.035, f"{name} rewrote the first {v['n_requested']} abstracts of the 2020 sample in its own words, "
             f"keeping every fact;\nAnthropic's safety filter declined {v['n_refused']}. "
             f"Both bars show the same {n} abstracts.",
             fontsize=7.5, color=MUTE, ha="left", va="bottom", linespacing=1.4)
    save(fig, stem)
    plt.close(fig)


def figure3(v, stem):
    plt = plt_setup()
    from matplotlib.patches import Patch
    name = v["generator"]["model_name"]
    bars = fullgen_bars(v)
    ys = [3.45, 2.0, 1.0, 0.0]        # a gap between the all-controls bar and the three subgroups
    fig = plt.figure(figsize=(7.0, 4.0))
    ax = fig.add_axes([0.36, 0.27, 0.59, 0.47])
    labels = []
    for y, (key, c) in zip(ys, bars):
        n = c["AI"] + c["Mixed"] + c["Human"]
        labels.append(f"All controls\n({n} abstracts)" if key == "all"
                      else f"Source labeled\n{SOURCE_LABEL[key]} ({n})")
        left = 0.0
        for k, col, dark in (("AI", C_AI, True), ("Mixed", C_MIXED, True), ("Human", C_HUMAN, False)):
            w = c[k] / n
            if not w:
                continue
            ax.barh(y, w, 0.58, left=left, color=col, linewidth=0, zorder=2)
            if w >= 0.08:
                ax.text(left + w / 2, y, f"{100 * w:.1f}%", ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if dark else INK, zorder=3)
            left += w
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, ys[0] + 0.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=C_AI, label="AI-written"), Patch(color=C_MIXED, label="Partly AI-written"),
                       Patch(color=C_HUMAN, label="Human-written")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=8.5, handlelength=1.4)
    fig.text(0.03, 0.965, "How the detector labels abstracts written entirely by AI",
             fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.905, f"Detector: {MODEL}. Lower three bars: grouped by how the detector labeled the source abstract.",
             fontsize=8.5, color=MUTE, ha="left", va="top")
    fig.text(0.03, 0.03, f"{name} turned each sampled {v['source_year']} abstract into notes, then wrote a new "
             f"abstract from the notes alone.\n{v['n_candidates']} abstracts were tried; Anthropic's safety filter "
             f"declined {v['n_refused']}; the first {v['n_scored']} completed are the controls.",
             fontsize=7.5, color=MUTE, ha="left", va="bottom", linespacing=1.4)
    save(fig, stem)
    plt.close(fig)


# ---------------------------------------------------------------- main

def write_counts_outputs(rows, base, root):
    write_text(os.path.join(root, "results", "tables", "by_year.csv"), table_by_year(rows, base))
    v = load_validation()
    write_text(os.path.join(root, "results", "tables", "ai_rewrites.csv"), table_rewrites(v))
    figure1([r for r in rows if r["status"] != "pending"], base,
            os.path.join(root, "results", "figures", "figure1_flagged_by_year"))
    figure2(v, os.path.join(root, "results", "figures", "figure2_ai_rewrites"))
    c = load_fullgen()
    write_text(os.path.join(root, "results", "tables", "ai_written_controls.csv"), table_fullgen(c))
    figure3(c, os.path.join(root, "results", "figures", "figure3_ai_written_controls"))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--write-expected", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        rows, base, _ = year_rows(counts_from_manifest())
        exp = json.load(open(EXPECTED, encoding="utf-8"))["sha256"]
        with tempfile.TemporaryDirectory() as d:
            write_counts_outputs(rows, base, d)
            bad = 0
            for rel in SELFTEST_OUTPUTS:
                ok = sha_file(os.path.join(d, rel)) == exp[rel]
                print(f"{rel}: {'IDENTICAL to the published build' if ok else 'DIFFERENT'}")
                bad += not ok
        if bad:
            sys.exit("FIGURE SELFTEST FAILED")
        print("FIGURE SELFTEST PASSED (figures and their tables rebuilt from the published counts)")
        return

    if a.write_expected:
        d = {"about": "SHA-256 of the analysis outputs built from the study's Pangram responses. "
                      "`reproduce.py analyze` compares a run with these; `selftest` rebuilds the figures "
                      "from the published counts and compares them too. PDF and PNG copies are not compared.",
             "sha256": {rel: sha_file(os.path.join(ROOT, rel)) for rel in OUTPUTS}}
        write_text(EXPECTED, json.dumps(d, indent=2) + "\n")
        print(f"-> {EXPECTED}")
        return

    by_year, year_of = load_labels()
    if not by_year:
        sys.exit("no results yet: run `reproduce.py rehydrate` (or `score`) first")
    raw = load_raw(year_of)
    check_raw(raw, by_year, year_of)
    rows, base, n_tests = year_rows(counts_of(by_year))
    write_counts_outputs(rows, base, ROOT)
    terms = build_terms(raw)
    write_text(os.path.join(ROOT, "results", "tables", "terms.csv"), table_terms(terms))
    write_text(os.path.join(ROOT, "results", "tables", "term_tests.json"), term_tests_json(terms))
    for r in rows:
        if r["status"] == "scored":
            extra = f", q = {r['q']:.3g}" if "q" in r else (" (baseline)" if r.get("is_base") else "")
            print(f"{r['year']}: {r['flag']}/{r['n']} flagged ({r['ai']} AI, {r['mixed']} Mixed){extra}")
        else:
            print(f"{r['year']}: {r['note']}")
    print(f"terms: {len(terms['words'])} words, {len(terms['phrases'])} phrases listed "
          f"(of {terms['rule']['n_year_tests']:,} / {terms['rule']['n_hl_tests']:,} tested)")
    for rel in ("results/tables", "results/figures"):
        for f in sorted(os.listdir(os.path.join(ROOT, rel))):
            print("->", f"{rel}/{f}")


if __name__ == "__main__":
    main()
