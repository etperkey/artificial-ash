"""Compare a run's per-year Pangram labels with the published results.

manifests/results_summary.csv holds, for each year, the number of abstracts scored and
how many Pangram labeled AI, Mixed or Human, plus the Benjamini-Hochberg q-value of the
Fisher test against pooled 2020-2021. This re-implements those statistics independently
of analysis.py.

Usage:
  python -I check_results.py --check      # results/*.scores.jsonl vs the manifest (exit 1 if different)
  python -I check_results.py --write      # maintainer: write the manifest from results/
  python -I check_results.py --selftest   # no data needed: recompute q-values from the published counts
"""
import argparse
import csv
import glob
import json
import os
import sys

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MANIFEST = os.path.join(ROOT, "manifests", "results_summary.csv")
BASELINE = (2020, 2021)
FIELDS = ("year", "n", "ai", "mixed", "human", "flagged", "q_vs_baseline")


def counts_from_results(results):
    by = {}
    for p in glob.glob(os.path.join(results, "*.scores.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("stage") != "STAGE_SUCCESS":
                continue
            c = by.setdefault(r["year"], {"n": 0, "ai": 0, "mixed": 0, "human": 0})
            c["n"] += 1
            c[{"AI": "ai", "Mixed": "mixed"}.get(r["prediction_short"], "human")] += 1
    return dict(sorted(by.items()))


def with_q(by):
    rows = [{"year": y, **c, "flagged": c["ai"] + c["mixed"]} for y, c in by.items()]
    base = [r for r in rows if r["year"] in BASELINE]
    bn, bk = sum(r["n"] for r in base), sum(r["flagged"] for r in base)
    comp = [r for r in rows if r["year"] not in BASELINE]
    ps = [fisher_exact([[r["flagged"], r["n"] - r["flagged"]], [bk, bn - bk]])[1] for r in comp]
    qs = multipletests(ps, method="fdr_bh")[1] if ps else []
    for r in rows:
        r["q_vs_baseline"] = ""
    for r, q in zip(comp, qs):
        r["q_vs_baseline"] = repr(float(q))
    return rows


def read_manifest():
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return [{k: (int(v) if k not in ("q_vs_baseline",) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def same_q(a, b):
    if a == "" or b == "":
        return a == b
    a, b = float(a), float(b)
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)


def compare(got, want):
    ok = len(got) == len(want)
    for g, w in zip(got, want):
        match = all(g[k] == w[k] for k in FIELDS[:-1]) and same_q(g["q_vs_baseline"], w["q_vs_baseline"])
        ok &= match
        print(f"{g['year']}: {g['flagged']}/{g['n']} flagged ({g['ai']} AI, {g['mixed']} Mixed), "
              f"q = {g['q_vs_baseline'] or '-'}  -> {'MATCH' if match else 'DIFFERENT from ' + str(w)}")
    return ok


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    if a.write:
        rows = with_q(counts_from_results(a.results))
        with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {MANIFEST}")
        return
    want = read_manifest()
    if a.selftest:
        got = with_q({r["year"]: {k: r[k] for k in ("n", "ai", "mixed", "human")} for r in want})
        ok = compare(got, want)
        print("SELFTEST PASSED: published q-values recomputed" if ok else "SELFTEST FAILED")
    else:
        got = with_q(counts_from_results(a.results))
        if not got:
            sys.exit("no results/*.scores.jsonl (run rehydrate or score first)")
        ok = compare(got, want)
        print("RESULTS MATCH the published counts" if ok else "RESULTS DIFFER from the published counts")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
