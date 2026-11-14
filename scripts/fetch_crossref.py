"""Pull every ASH annual-meeting abstract for one year from Crossref.

ASH abstracts are published as Blood (ISSN 0006-4971) "Supplement 1" of the
year's second volume. Crossref carries the full abstract text as JATS XML.

Usage:  python -I fetch_crossref.py 2021 [--out DIR]
Output: <out>/crossref_<year>.jsonl  (one raw record per abstract, atomic write)
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ISSN = "0006-4971"
# Blood volume carrying the ASH supplement: 2019=134, 2020=136, ... (2 vols/yr)
VOLUME = {y: str(134 + 2 * (y - 2019)) for y in range(2019, 2031)}
FIELDS = "DOI,title,abstract,author,page,volume,issue,published,type,created"
UA = {"User-Agent": "ash-ai-audit/0.1 (academic research script)"}


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # network hiccup / 429 / 5xx
            wait = 5 * (i + 1)
            print(f"  retry {i + 1}/{tries} after error: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    a = ap.parse_args()
    vol = VOLUME[a.year]

    base = f"https://api.crossref.org/journals/{ISSN}/works"
    params = {
        "filter": f"from-pub-date:{a.year}-01-01,until-pub-date:{a.year + 1}-03-31",
        "rows": "1000",
        "select": FIELDS,
        "cursor": "*",
    }
    kept, seen, total = [], set(), None
    while True:
        msg = get(base + "?" + urllib.parse.urlencode(params))["message"]
        total = msg.get("total-results", total)
        items = msg.get("items", [])
        for it in items:
            if it.get("volume") == vol and (it.get("issue") or "").startswith("Supplement") \
                    and it["DOI"] not in seen:
                seen.add(it["DOI"])
                kept.append(it)
        print(f"  page: {len(items)} items, kept so far {len(kept)} / scanned window total {total}")
        nxt = msg.get("next-cursor")
        if not items or not nxt:
            break
        params["cursor"] = nxt
        time.sleep(1)  # be polite to the public API

    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"crossref_{a.year}.jsonl")
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, dest)
    n_abs = sum(1 for it in kept if it.get("abstract"))
    print(f"{a.year}: vol {vol} supplement records = {len(kept)}, with abstract text = {n_abs} -> {dest}")


if __name__ == "__main__":
    main()
