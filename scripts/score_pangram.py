"""Score abstracts with the Pangram text API; resumable, budget-capped, fully logged.

Every raw API response is kept (results/<name>.raw.jsonl) so the analysis can be
re-derived without re-paying. Already-scored ids are skipped on re-run.

Usage:
  python -I score_pangram.py INPUT.jsonl --estimate              # cost only, no calls
  python -I score_pangram.py INPUT.jsonl --limit 20 --max-usd 8  # first 20 rows
  python -I score_pangram.py INPUT.jsonl --max-usd 200 --workers 4

Key: env PANGRAM_API_KEY, else ../.pangram_key (outside any git repo).
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
API = "https://text.external-api.pangram.com"
MODEL = "pangram-4.1"          # pinned for the whole study
USD_PER_100_WORDS = 0.05       # list price, Pangram 4.1 text (2026-11)
KEEP = ("version", "headline", "prediction_short", "fraction_ai", "fraction_ai_assisted",
        "fraction_human", "num_ai_segments", "num_ai_assisted_segments", "num_human_segments")
lock = threading.Lock()


def api_key():
    k = os.environ.get("PANGRAM_API_KEY")
    if not k:
        with open(os.path.join(ROOT, ".pangram_key"), encoding="utf-8") as f:
            k = f.read().strip()
    return k


class Fatal(Exception):
    pass


def call(method, path, key, body=None, tries=6):
    data = json.dumps(body).encode() if body is not None else None
    for i in range(tries):
        req = urllib.request.Request(API + path, data=data, method=method,
                                     headers={"x-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (401, 402, 403, 413, 422):  # auth / out of credits / bad request: stop, don't burn retries
                raise Fatal(f"HTTP {e.code}: {msg}")
            wait = min(60, 3 * 2 ** i)
            print(f"  HTTP {e.code} ({msg[:80]}), retry in {wait}s", file=sys.stderr)
        except Exception as e:
            wait = min(60, 3 * 2 ** i)
            print(f"  {e}, retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"{method} {path} failed after {tries} tries")


def score_one(row, key):
    task = call("POST", "/task", key, {"text": row["text"], "model": MODEL})
    tid = task["task_id"]
    for _ in range(240):
        res = call("GET", f"/task/{tid}", key)
        if res.get("stage") in ("STAGE_SUCCESS", "STAGE_FAILED"):
            return tid, res
        time.sleep(1.5)
    raise RuntimeError(f"task {tid} did not finish")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--name", help="results basename (default: input basename)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    name = a.name or os.path.splitext(os.path.basename(a.input))[0]
    out = os.path.join(ROOT, "results", f"{name}.scores.jsonl")
    raw = os.path.join(ROOT, "results", f"{name}.raw.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    scope = rows[: a.limit] if a.limit is not None else rows  # --limit N = first N rows of the file
    todo = [r for r in scope if r["id"] not in done]
    words = sum(r["n_words"] for r in todo)
    usd = words / 100 * USD_PER_100_WORDS
    print(f"{name}: {len(rows)} rows, {len(done)} already scored, {len(todo)} to score, "
          f"{words:,} words = ${usd:,.2f} at list price")
    if a.estimate or not todo:
        return
    if usd > a.max_usd:
        sys.exit(f"Refusing: ${usd:,.2f} exceeds --max-usd {a.max_usd}. Raise the cap deliberately.")

    key = api_key()
    n_ok = n_fail = 0
    with open(out, "a", encoding="utf-8") as fo, open(raw, "a", encoding="utf-8") as fr, \
            cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(score_one, r, key): r for r in todo}
        try:
            for fut in cf.as_completed(futs):
                r = futs[fut]
                try:
                    tid, res = fut.result()
                except Fatal as e:
                    print(f"FATAL on {r['id']}: {e}\nStopping; finished rows are saved. Re-run to resume.")
                    for f2 in futs:
                        f2.cancel()
                    break
                except Exception as e:
                    n_fail += 1
                    print(f"  failed {r['id']}: {e}", file=sys.stderr)
                    continue
                rec = {"id": r["id"], "year": r["year"], "task_id": tid, "model": MODEL,
                       "stage": res.get("stage"), "n_words": r["n_words"],
                       **{k: res.get(k) for k in KEEP}}
                with lock:
                    fr.write(json.dumps({"id": r["id"], "task_id": tid, "response": res}, ensure_ascii=False) + "\n")
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fo.flush(); fr.flush()
                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"  {n_ok}/{len(todo)} scored")
        except KeyboardInterrupt:
            print("Interrupted; finished rows are saved.")
    print(f"done: {n_ok} scored, {n_fail} failed -> {out}")


if __name__ == "__main__":
    main()
