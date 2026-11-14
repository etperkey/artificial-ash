"""One command for every step of the ASH abstract AI-text audit.

  python -I reproduce.py data        download ASH abstracts from Crossref, clean them, draw the samples
  python -I reproduce.py verify      prove the downloaded abstracts are the ones the study analyzed
  python -I reproduce.py rehydrate   rebuild the Pangram results from the stripped replay files
  python -I reproduce.py score       re-score the samples yourself (needs a paid PANGRAM_API_KEY)
  python -I reproduce.py analyze     statistics, the dashboard page, and a check against the published numbers
  python -I reproduce.py all         data + verify + rehydrate + analyze
  python -I reproduce.py selftest    quick offline check of the software environment

Options: --replay DIR (default replay/), --max-usd N for `score`, --years 2020 2021 ...
Set ASH_BUILD_DATE=YYYY-MM-DD to stamp the page with a fixed date (needed for a byte-for-byte match).
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(ROOT, "scripts")
YEARS = (2020, 2021, 2023, 2024, 2025)   # 2022: Crossref has no abstract text
SAMPLE_N, SEED = 500, 20261113           # random order; the first SCORED_N rows were scored
SCORED_N = 250
USD_PER_100_WORDS = 0.05                 # Pangram list price (2026-11), as in score_pangram.py


def shown(c):
    c = str(c)
    if not os.path.isabs(c):
        return c
    try:
        return os.path.relpath(c, ROOT)
    except ValueError:  # Windows: a path on another drive
        return c


def run(*args, env=None):
    cmd = [sys.executable, "-I", *map(str, args)]
    print("$", " ".join(shown(c) for c in cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode:
        sys.exit(r.returncode)


def sample_path(y):
    return os.path.join(ROOT, "data", "samples", f"sample_{y}_n{SAMPLE_N}_seed{SEED}.jsonl")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def step_data(years):
    os.makedirs(os.path.join(ROOT, "data", "samples"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "clean"), exist_ok=True)
    for y in years:
        run(os.path.join(S, "fetch_crossref.py"), y)
        run(os.path.join(S, "clean.py"), y)
        run(os.path.join(S, "sample.py"), y, SAMPLE_N, "--seed", SEED)


def step_score(years, max_usd):
    if os.path.exists(os.path.join(ROOT, "results", "REHYDRATED")):
        sys.exit("results/ holds the study's rehydrated Pangram results; scoring would skip every abstract\n"
                 "as already done. Move or delete results/ first to score the samples yourself.")
    if not os.environ.get("PANGRAM_API_KEY") and not os.path.exists(os.path.join(ROOT, ".pangram_key")):
        sys.exit("score needs a Pangram API key: set PANGRAM_API_KEY (scoring costs money; see README)")
    # --max-usd caps the TOTAL across years; each year gets exactly its own estimate
    cost = {}
    for y in years:
        rows = [json.loads(l) for l in open(sample_path(y), encoding="utf-8")][:SCORED_N]
        cost[y] = sum(r["n_words"] for r in rows) / 100 * USD_PER_100_WORDS
    total = sum(cost.values())
    print(f"scoring {len(years)} years x {SCORED_N} abstracts: about ${total:,.2f} at list price "
          f"(already-scored abstracts are skipped)")
    if total > max_usd:
        sys.exit(f"Refusing: up to ${total:,.2f} exceeds --max-usd {max_usd}. Raise the cap deliberately.")
    for y in years:
        run(os.path.join(S, "score_pangram.py"), sample_path(y), "--limit", SCORED_N,
            "--max-usd", f"{cost[y] + 0.01:.2f}")


def step_analyze():
    scores = sorted(glob.glob(os.path.join(ROOT, "results", "*.scores.jsonl")))
    if not scores:
        sys.exit("no results yet: run `reproduce.py rehydrate` (or `score`) first")
    run(os.path.join(S, "summarize.py"), *scores)
    run(os.path.join(S, "build_dashboard.py"))
    run(os.path.join(S, "check_results.py"), "--check")
    exp_path = os.path.join(ROOT, "manifests", "expected_outputs.json")
    if os.path.exists(exp_path):
        exp = json.load(open(exp_path, encoding="utf-8"))
        if os.environ.get("ASH_BUILD_DATE") != exp["build_date"]:
            print(f"(page not compared byte for byte: set ASH_BUILD_DATE={exp['build_date']} to do that)")
            return
        bad = 0
        for rel, want in exp["sha256"].items():
            got = sha_file(os.path.join(ROOT, rel))
            print(f"{rel}: {'IDENTICAL to the published build' if got == want else 'DIFFERENT (' + got[:12] + ')'}")
            bad += got != want
        if bad:
            sys.exit(1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("data", "verify", "rehydrate", "score", "analyze", "all", "selftest"))
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--replay", default=os.path.join(ROOT, "replay"))
    ap.add_argument("--max-usd", type=float, default=0.0)
    a = ap.parse_args()
    if a.step in ("data", "all"):
        step_data(a.years)
    if a.step in ("verify", "all"):
        run(os.path.join(S, "fingerprint.py"), "--check")
    if a.step in ("rehydrate", "all"):
        run(os.path.join(S, "pangram_replay.py"), "rehydrate", "--replay", a.replay)
    if a.step == "score":
        step_score(a.years, a.max_usd)
    if a.step in ("analyze", "all"):
        step_analyze()
    if a.step == "selftest":
        run(os.path.join(S, "check_results.py"), "--selftest")
        run(os.path.join(ROOT, "tests", "test_replay.py"))


if __name__ == "__main__":
    main()
