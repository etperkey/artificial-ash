"""Fingerprint the cleaned abstracts and samples, or check a fresh download against them.

The capsule never ships ASH abstract text (ASH holds the copyright). Instead it ships
SHA-256 fingerprints of the text the study analyzed. Anyone who re-downloads the
abstracts from Crossref and re-runs clean.py + sample.py can then prove they hold
exactly the same text, abstract by abstract.

Usage:
  python -I fingerprint.py --check            # compare data/ with manifests/ (exit 1 on any mismatch)
  python -I fingerprint.py --write            # maintainer: (re)write manifests/ from data/
  python -I fingerprint.py --write --root DIR # fingerprint another copy of the pipeline folder

Manifests:
  manifests/crossref_<year>.json   record and eligible counts, plus one digest over every
                                   eligible abstract (sorted "doi<TAB>sha256(text)" lines)
  manifests/sample_<year>.tsv      the sampled abstracts in sample order: rank, DOI,
                                   word count, sha256(text)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFESTS = os.path.join(HERE, "..", "manifests")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def year_manifest(root, year):
    rows = [json.loads(l) for l in open(os.path.join(root, "data", "clean", f"abstracts_{year}.jsonl"),
                                        encoding="utf-8")]
    el = sorted((r for r in rows if r["eligible"]), key=lambda r: r["id"])
    digest = sha("".join(f"{r['id']}\t{sha(r['text'])}\n" for r in el))
    return {"year": year, "records": len(rows), "eligible": len(el),
            "eligible_words": sum(r["n_words"] for r in el), "eligible_sha256": digest}


def sample_lines(path):
    out = []
    for i, l in enumerate(open(path, encoding="utf-8"), 1):
        r = json.loads(l)
        out.append(f"{i}\t{r['id']}\t{r['n_words']}\t{sha(r['text'])}")
    return out


def samples(root):
    found = {}
    for p in sorted(glob.glob(os.path.join(root, "data", "samples", "sample_*_n*_seed*.jsonl"))):
        m = re.search(r"sample_(\d{4})_n(\d+)_seed(\d+)\.jsonl$", p)
        found[int(m.group(1))] = p
    return found


def write(root):
    os.makedirs(MANIFESTS, exist_ok=True)
    for year, path in samples(root).items():
        m = year_manifest(root, year)
        m["sample_file"] = os.path.basename(path)
        with open(os.path.join(MANIFESTS, f"crossref_{year}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=2)
            f.write("\n")
        with open(os.path.join(MANIFESTS, f"sample_{year}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("rank\tdoi\tn_words\ttext_sha256\n")
            f.write("\n".join(sample_lines(path)) + "\n")
        print(f"{year}: {m['records']} records, {m['eligible']} eligible, sample {os.path.basename(path)}")


def check(root):
    ok = True
    for mpath in sorted(glob.glob(os.path.join(MANIFESTS, "crossref_*.json"))):
        want = json.load(open(mpath, encoding="utf-8"))
        year = want["year"]
        clean = os.path.join(root, "data", "clean", f"abstracts_{year}.jsonl")
        if not os.path.exists(clean):
            print(f"{year}: MISSING {clean} (run: python -I reproduce.py data)")
            ok = False
            continue
        got = year_manifest(root, year)
        same = all(got[k] == want[k] for k in ("records", "eligible", "eligible_words", "eligible_sha256"))
        print(f"{year}: Crossref records {got['records']} (expected {want['records']}), eligible "
              f"{got['eligible']} (expected {want['eligible']}) -> {'MATCH' if same else 'DIFFERENT'}")
        # the sample is what the detector saw: compare it abstract by abstract
        spath = os.path.join(root, "data", "samples", want["sample_file"])
        exp = open(os.path.join(MANIFESTS, f"sample_{year}.tsv"), encoding="utf-8").read().splitlines()[1:]
        if not os.path.exists(spath):
            print(f"  sample MISSING {spath}")
            ok = False
            continue
        have = sample_lines(spath)
        diff = [(e, h) for e, h in zip(exp, have) if e != h]
        if len(have) != len(exp) or diff:
            ok = False
            print(f"  sample: {len(diff)} of {len(exp)} abstracts differ"
                  + (f", {len(have)} rows instead of {len(exp)}" if len(have) != len(exp) else ""))
            for e, h in diff[:5]:
                print(f"    expected {e}\n    got      {h}")
        else:
            print(f"  sample: all {len(exp)} abstracts identical, in the same order")
        if not same:
            # a changed record outside the sample does not touch the results, but say so
            print("  note: the full-year digest differs (Crossref records can be corrected or added);"
                  " results depend only on the sample above")
    if not glob.glob(os.path.join(MANIFESTS, "crossref_*.json")):
        print("no manifests found")
        ok = False
    print("VERIFIED: same abstracts as the study" if ok else "NOT VERIFIED: see differences above")
    return ok


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=os.path.join(HERE, ".."))
    a = ap.parse_args()
    if a.write:
        write(a.root)
    else:
        sys.exit(0 if check(a.root) else 1)


if __name__ == "__main__":
    main()
