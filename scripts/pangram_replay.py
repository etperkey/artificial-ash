"""Share Pangram results without sharing ASH's abstract text, and put the text back locally.

A raw Pangram response repeats the abstract it scored (the whole text, and each
judged window's text). ASH owns those abstracts, so the capsule cannot redistribute
them. `strip` removes every piece of text and keeps what is needed to restore it:

  * text_sha256 for the scored text and for every window, to verify the restoration
  * dropped: positions of the characters Pangram removed from what it was sent
    (it collapses blank lines and drops a few invisible characters), so the exact
    text it scored can be rebuilt from the cleaned abstract
  * window start_index / end_index, which Pangram already reports

`fetch` downloads the stripped files the study published (eperkey.org/artificial_ash/capsule/replay/)
and keeps each only if its SHA-256 matches manifests/replay_sha256.txt.

`rehydrate` takes the stripped files plus your own data/samples/ (rebuilt from Crossref
by `reproduce.py data`) and writes results/*.raw.jsonl and results/*.scores.jsonl exactly
as score_pangram.py would have, refusing if any hash does not match.

Usage:
  python -I pangram_replay.py fetch      --replay replay                      # anyone
  python -I pangram_replay.py rehydrate  --replay replay   --results results  # anyone with the replay files
  python -I pangram_replay.py strip      --results results --out replay       # maintainer (also writes the checksums)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODEL = "pangram-4.1"
URL = "https://eperkey.org/artificial_ash/capsule/replay/"   # where the study publishes the replay files
SUMS = os.path.join(ROOT, "manifests", "replay_sha256.txt")
KEEP = ("version", "headline", "prediction_short", "fraction_ai", "fraction_ai_assisted",
        "fraction_human", "num_ai_segments", "num_ai_assisted_segments", "num_human_segments")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dropped_positions(sent, scored):
    """Indices of `sent` whose removal gives `scored`, or None if it is not a pure deletion."""
    i = j = 0
    out = []
    while i < len(sent):
        if j < len(scored) and sent[i] == scored[j]:
            j += 1
        else:
            out.append(i)
        i += 1
    return out if j == len(scored) else None


def apply_drops(sent, drops):
    gone = set(drops)
    return "".join(ch for k, ch in enumerate(sent) if k not in gone)


def load_samples(root):
    rows = {}
    for p in glob.glob(os.path.join(root, "data", "samples", "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def strip(results, out, root):
    samples = load_samples(root)
    os.makedirs(out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(results, "*.raw.jsonl"))):
        name = os.path.basename(p)[: -len(".raw.jsonl")]
        recs = []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            res = dict(r["response"])
            sent = samples[r["id"]]["text"]
            scored = res.pop("text", None)
            if scored is not None:
                drops = dropped_positions(sent, scored)
                if drops is None or apply_drops(sent, drops) != scored:
                    raise SystemExit(f"{r['id']}: Pangram's text is not a deletion-only edit of the sample text")
                res["text_sha256"] = sha(scored)
                res["dropped"] = drops
            wins = []
            for w in res.get("windows") or []:
                w = dict(w)
                wt = w.pop("text")
                if scored[w["start_index"]:w["end_index"]] != wt:
                    raise SystemExit(f"{r['id']}: a window's offsets do not match its text")
                w["text_sha256"] = sha(wt)
                wins.append(w)
            if "windows" in res:
                res["windows"] = wins
            recs.append({"id": r["id"], "task_id": r["task_id"], "response": res})
        dest = os.path.join(out, f"{name}.replay.jsonl")
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        blob = open(dest, encoding="utf-8").read()
        for r in recs:  # belt and braces: no abstract text may survive
            if samples[r["id"]]["text"][:60] in blob:
                raise SystemExit(f"abstract text leaked into {dest}")
        print(f"{name}: {len(recs)} responses stripped -> {dest}")
    write_sums(out)


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_sums(replay):
    """Maintainer: record the SHA-256 of each replay file (sha256sum format) for `fetch`."""
    lines = [f"{file_sha(p)}  {os.path.basename(p)}\n"
             for p in sorted(glob.glob(os.path.join(replay, "*.replay.jsonl")))]
    with open(SUMS + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    os.replace(SUMS + ".tmp", SUMS)
    print(f"{len(lines)} checksums -> {SUMS}")


def fetch(replay, url):
    """Download the published replay files and keep each only if its SHA-256 matches the manifest."""
    import urllib.request
    want = {}
    for line in open(SUMS, encoding="utf-8"):
        if line.strip():
            h, name = line.split()
            want[name.lstrip("*")] = h
    os.makedirs(replay, exist_ok=True)
    for name, h in want.items():
        dest = os.path.join(replay, name)
        if os.path.exists(dest) and file_sha(dest) == h:
            print(f"{name}: already here, checksum OK")
            continue
        req = urllib.request.Request(url + name, headers={"User-Agent": "artificial-ash-capsule"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        got = hashlib.sha256(body).hexdigest()
        if got != h:
            raise SystemExit(f"{name}: checksum {got[:12]}… does not match the manifest ({h[:12]}…); not saved")
        with open(dest + ".tmp", "wb") as f:
            f.write(body)
        os.replace(dest + ".tmp", dest)
        print(f"{name}: downloaded, checksum OK ({len(body) / 1e3:.0f} kB)")
    print(f"REPLAY FILES OK: {len(want)} files match manifests/replay_sha256.txt")


def rehydrate(replay, results, root, expect):
    samples = load_samples(root)
    if not samples:
        raise SystemExit("no data/samples found; run `python -I reproduce.py data` first")
    os.makedirs(results, exist_ok=True)
    files = sorted(glob.glob(os.path.join(replay, "*.replay.jsonl")))
    if not files:
        raise SystemExit(f"no *.replay.jsonl files in {replay}")
    total = 0
    per_year = {}
    for p in files:
        name = os.path.basename(p)[: -len(".replay.jsonl")]
        raw_lines, score_lines = [], []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            row = samples.get(r["id"])
            if row is None:
                raise SystemExit(f"{r['id']} is not in your data/samples; rebuild them with reproduce.py data")
            res = dict(r["response"])
            want = res.pop("text_sha256", None)
            drops = res.pop("dropped", None)
            if want is not None:
                scored = apply_drops(row["text"], drops)
                if sha(scored) != want:
                    raise SystemExit(f"{r['id']}: your abstract text differs from the text Pangram scored")
                wins = []
                for w in res.get("windows") or []:
                    w = dict(w)
                    wh = w.pop("text_sha256")
                    wt = scored[w["start_index"]:w["end_index"]]
                    if sha(wt) != wh:
                        raise SystemExit(f"{r['id']}: window text does not match")
                    wins.append(rebuild_window(w, wt))
                full = {}
                for k, v in res.items():  # put "text" back where Pangram had it (after "stage")
                    full[k] = v
                    if k == "stage":
                        full["text"] = scored
                if "windows" in res:
                    full["windows"] = wins
                res = full
            raw_lines.append(json.dumps({"id": r["id"], "task_id": r["task_id"], "response": res},
                                        ensure_ascii=False))
            rec = {"id": r["id"], "year": row["year"], "task_id": r["task_id"], "model": MODEL,
                   "stage": res.get("stage"), "n_words": row["n_words"], **{k: res.get(k) for k in KEEP}}
            score_lines.append(json.dumps(rec, ensure_ascii=False))
            per_year[row["year"]] = per_year.get(row["year"], 0) + 1
        for suffix, lines in (("raw", raw_lines), ("scores", score_lines)):
            dest = os.path.join(results, f"{name}.{suffix}.jsonl")
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
        total += len(raw_lines)
        print(f"{name}: {len(raw_lines)} Pangram responses restored and verified")
    # every year the study scored must come back complete
    man = expect
    if man != "none":
        want = {int(r["year"]): int(r["n"]) for r in csv.DictReader(open(man, encoding="utf-8", newline=""))}
        if per_year != want:
            raise SystemExit(f"incomplete replay set: restored {per_year}, the study scored {want}")
    with open(os.path.join(results, "REHYDRATED"), "w", encoding="utf-8", newline="\n") as f:
        f.write("These results were rebuilt from the study's replay files, not scored anew.\n")
    print(f"REHYDRATED: {total} responses, every text hash matched")


def rebuild_window(w, text):
    """Pangram lists "text" first in each window; restore that key order."""
    out = {"text": text}
    out.update(w)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fetch", "strip", "rehydrate"))
    ap.add_argument("--url", default=URL, help="fetch: base URL of the published replay files")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--replay", default=os.path.join(ROOT, "replay"))
    ap.add_argument("--out", default=os.path.join(ROOT, "replay"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--expect", default=os.path.join(ROOT, "manifests", "results_summary.csv"),
                    help="per-year counts the restored set must match ('none' to skip; tests only)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.replay, a.url)
    elif a.cmd == "strip":
        strip(a.results, a.out, a.root)
    else:
        rehydrate(a.replay, a.results, a.root, a.expect)


if __name__ == "__main__":
    main()
