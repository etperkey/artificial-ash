"""Draw a reproducible simple random sample of eligible abstracts for one year.

Usage:  python -I sample.py 2021 500 [--seed 20261113]
Output: data/samples/sample_<year>_n<N>_seed<S>.jsonl (in random order, so the
first k rows of the file are themselves a random sample -> pilots are not wasted).
"""
import argparse
import json
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

ap = argparse.ArgumentParser()
ap.add_argument("year", type=int)
ap.add_argument("n", type=int)
ap.add_argument("--seed", type=int, default=20261113)
a = ap.parse_args()

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "clean", f"abstracts_{a.year}.jsonl"), encoding="utf-8")]
pool = sorted((r for r in rows if r["eligible"]), key=lambda r: r["id"])  # sort first: order-independent of download
picked = random.Random(a.seed).sample(pool, a.n)

dest = os.path.join(ROOT, "data", "samples", f"sample_{a.year}_n{a.n}_seed{a.seed}.jsonl")
tmp = dest + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(tmp, dest)
words = sum(r["n_words"] for r in picked)
print(f"{a.year}: sampled {a.n} of {len(pool)} eligible (seed {a.seed}) -> {dest}")
print(f"  words {words:,}  est. cost ${words / 100 * 0.05:,.2f} list / ${words / 100 * 0.04:,.2f} bulk")
