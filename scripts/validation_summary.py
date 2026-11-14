"""Maintainer: summarize the AI-rewrite validation run into manifests/validation_rewrite.json.

The rewrites are derivatives of ASH's copyrighted abstracts, so they are not published; this
manifest holds only what the public page shows (counts, the prompt, settings, refusals).
Figure 2 on the page is built from it.

Usage:  python -I validation_summary.py VALIDATION_DIR
        (VALIDATION_DIR holds api_batch.json, api_results.jsonl, index.jsonl, qc_2020.tsv and
         results/rewrite_claude_opus55_2020.{scores,raw}.jsonl)
"""
import collections
import json
import os
import sys

from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
NAME = "rewrite_claude_opus55_2020"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    d = sys.argv[1]
    batch = json.load(open(os.path.join(d, "api_batch.json"), encoding="utf-8"))
    api = jl(os.path.join(d, "api_results.jsonl"))
    index = {x["id"]: x["k"] for x in jl(os.path.join(d, "index.jsonl"))}
    scores = [r for r in jl(os.path.join(d, "results", f"{NAME}.scores.jsonl")) if r["stage"] == "STAGE_SUCCESS"]
    raw = {x["id"]: x["response"] for x in jl(os.path.join(d, "results", f"{NAME}.raw.jsonl"))}
    kept = {}
    for line in open(os.path.join(d, "qc_2020.tsv"), encoding="utf-8").read().splitlines()[1:]:
        f = line.split("\t")
        kept[f[0]] = float(f[4])
    # the same abstracts as originally written, from the study's own results
    orig = {}
    for p in sorted(os.listdir(os.path.join(ROOT, "results"))):
        if p.startswith("sample_2020") and p.endswith(".scores.jsonl"):
            for r in jl(os.path.join(ROOT, "results", p)):
                orig[r["id"]] = r["prediction_short"]

    refused = [r for r in api if r.get("stop_reason") == "refusal"]
    labels = collections.Counter(r["prediction_short"] for r in scores)
    src_ids = [r["id"].split("#")[0] for r in scores]
    olab = collections.Counter(orig[i] for i in src_ids)
    n = len(scores)
    flag = labels["AI"] + labels["Mixed"]
    lo, hi = proportion_confint(flag, n, method="wilson")
    share = sorted((r["fraction_ai"] or 0) + (r["fraction_ai_assisted"] or 0) for r in scores)
    wins = collections.Counter(w["label"] for res in raw.values() for w in res.get("windows", []))
    rho, p = spearmanr([kept[index[i]] for i in src_ids],
                       [(r["fraction_ai"] or 0) + (r["fraction_ai_assisted"] or 0) for r in scores])
    out = {
        "arm": NAME,
        "date": batch["created_at"][:10],
        "source": "the scored ASH 2020 sample abstracts, as cleaned for the study",
        "generator": {"model": batch["model"], "model_name": "Claude Opus 5.5", "interface": batch["endpoint"],
                      "effort": batch["effort"], "system_prompt": batch["system_prompt"],
                      "prompt": batch["prompt"], "prompt_sha256": batch["prompt_sha256"]},
        "detector": "pangram-4.1",
        "n_requested": batch["n_requests"],
        "n_refused": len(refused),
        "refusal_categories": dict(collections.Counter(r.get("refusal_category") for r in refused)),
        "n_scored": n,
        "labels": {k: labels.get(k, 0) for k in ("AI", "Mixed", "Human")},
        "originals_same_abstracts": {k: olab.get(k, 0) for k in ("AI", "Mixed", "Human")},
        "flagged_ci95": [round(lo, 4), round(hi, 4)],
        "share_ai_median": round(share[n // 2], 4),
        "windows": dict(wins),
        "spearman_wording_kept_vs_share_ai": round(float(rho), 3),
    }
    dest = os.path.join(ROOT, "manifests", "validation_rewrite.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: out[k] for k in ("n_requested", "n_refused", "n_scored", "labels",
                                         "originals_same_abstracts", "flagged_ci95")}))


if __name__ == "__main__":
    main()
