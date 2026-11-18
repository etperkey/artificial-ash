"""Maintainer: summarize the fully AI-written control run into manifests/validation_fullgen.json.

The controls are derived from ASH's copyrighted abstracts, so they are not published; this manifest
holds only what the public page shows (counts, the prompts, settings, refusals). Figure 3 is built
from it.

Usage:  python -I validation_fullgen_summary.py VALIDATION_DIR
        (VALIDATION_DIR holds PROMPT_NOTES.txt, PROMPT_WRITE.txt, index.jsonl, notes_results.jsonl,
         write_results.jsonl, fullgen_2025.jsonl, qc_2025.tsv and results/fullgen_2025.{scores,raw}.jsonl)
"""
import collections
import hashlib
import json
import os
import sys

from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
NAME = "fullgen_claude_opus55_2025"
YEAR = 2025
LABELS = ("AI", "Mixed", "Human")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def main():
    d = sys.argv[1]
    prompts = {n: open(os.path.join(d, n), encoding="utf-8").read().strip()
               for n in ("PROMPT_NOTES.txt", "PROMPT_WRITE.txt")}
    index = jl(os.path.join(d, "index.jsonl"))
    notes = jl(os.path.join(d, "notes_results.jsonl"))
    write = jl(os.path.join(d, "write_results.jsonl"))
    controls = {r["id"]: r for r in jl(os.path.join(d, "fullgen_2025.jsonl"))}
    scores = [r for r in jl(os.path.join(d, "results", "fullgen_2025.scores.jsonl")) if r["stage"] == "STAGE_SUCCESS"]
    raw = {x["id"]: x["response"] for x in jl(os.path.join(d, "results", "fullgen_2025.raw.jsonl"))}
    if {r["id"] for r in scores} != set(controls):
        sys.exit("the scored controls are not exactly the assembled controls")
    qc = {}
    for line in open(os.path.join(d, "qc_2025.tsv"), encoding="utf-8").read().splitlines()[1:]:
        f = line.split("\t")
        qc[f[0]] = (float(f[3]), float(f[4]))
    # the source abstracts as originally written, from the study's own results
    orig = {}
    for p in sorted(os.listdir(os.path.join(ROOT, "results"))):
        if p.startswith(f"sample_{YEAR}") and p.endswith(".scores.jsonl"):
            for r in jl(os.path.join(ROOT, "results", p)):
                orig[r["id"]] = r["prediction_short"]

    done = lambda r: r["stop_reason"] == "end_turn" and bool(r["text"].strip())
    refused_notes = [r for r in notes if r["stop_reason"] == "refusal"]
    refused_write = [r for r in write if r["stop_reason"] == "refusal"]
    other = [r for r in notes + write if not done(r) and r["stop_reason"] != "refusal"]
    n_complete = sum(done(r) for r in write)
    labels = collections.Counter(r["prediction_short"] for r in scores)
    by_src = {k: collections.Counter() for k in LABELS}
    for r in scores:
        by_src[orig[controls[r["id"]]["source_id"]]][r["prediction_short"]] += 1
    n = len(scores)
    lo, hi = proportion_confint(labels["AI"] + labels["Mixed"], n, method="wilson")
    share = [(r["fraction_ai"] or 0) + (r["fraction_ai_assisted"] or 0) for r in scores]
    ks = [controls[r["id"]]["k"] for r in scores]
    wins = collections.Counter(w["label"] for res in raw.values() for w in res.get("windows", []))
    models = sorted({r["model"] for r in notes + write})
    out = {
        "arm": NAME,
        "date": "2026-11-18",
        "source_year": YEAR,
        "source": f"the first {len(index)} abstracts of the ASH {YEAR} sample (random order), as cleaned for the study",
        "generator": {"model": models[0] if len(models) == 1 else models, "model_name": "Claude Opus 5.5",
                      "interface": "POST /v1/messages", "effort": "medium", "system_prompt": None,
                      "prompt_notes": prompts["PROMPT_NOTES.txt"], "prompt_write": prompts["PROMPT_WRITE.txt"],
                      "prompt_sha256": {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in prompts.items()}},
        "detector": "pangram-4.1",
        "n_candidates": len(index),
        "n_refused": len(refused_notes) + len(refused_write),
        "n_refused_notes_step": len(refused_notes),
        "n_refused_write_step": len(refused_write),
        "n_other_incomplete": len(other),
        "refusal_categories": dict(collections.Counter(str(r.get("refusal_category"))
                                                       for r in refused_notes + refused_write)),
        "n_complete": n_complete,
        "n_scored": n,
        "labels": {k: labels.get(k, 0) for k in LABELS},
        "flagged_ci95": [round(lo, 4), round(hi, 4)],
        "sources_same_abstracts": {k: sum(by_src[k].values()) for k in LABELS},
        "by_source_label": {k: {j: by_src[k].get(j, 0) for j in LABELS} for k in LABELS},
        "share_ai_median": round(median(share), 4),
        "windows": dict(wins),
        "length_ratio_median": round(median(qc[k][0] for k in ks), 2),
        "wording_kept_median": round(median(qc[k][1] for k in ks), 3),
        "wording_kept_max": round(max(qc[k][1] for k in ks), 3),
    }
    dest = os.path.join(ROOT, "manifests", "validation_fullgen.json")
    with open(dest + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(dest + ".tmp", dest)
    print(json.dumps({k: out[k] for k in ("n_candidates", "n_refused", "n_complete", "n_scored", "labels",
                                         "sources_same_abstracts", "by_source_label", "flagged_ci95")}))


if __name__ == "__main__":
    main()
