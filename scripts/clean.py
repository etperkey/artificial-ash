"""Turn raw Crossref ASH records into detector-ready prose plus covariates.

The SAME rules run on every year so that source formatting cannot masquerade
as a detector signal. Body text only (no title, authors, disclosures,
references, keyword lines or section-header labels).

Usage:  python -I clean.py 2021 [2025 ...]
Output: data/clean/abstracts_<year>.jsonl  +  a one-line summary per year
"""
import html
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MIN_WORDS = 75  # Pangram's documented minimum input length

DROP_SEC = re.compile(r"disclosure|off-?\s?label|funding|keyword|reference|acknowledg|conflict", re.I)
HEADER_WORDS = (r"background(?: and (?:significance|aims?|rationale))?|introduction|rationale|significance|"
                r"hypothesis|purpose|objectives?|aims?(?: and methods)?|study design(?: and methods)?|"
                r"(?:patients|materials|subjects),? and methods|methods?(?: and (?:patients|materials))?|"
                r"methodology|design|results?|findings|conclusions?|summary(?:/conclusions?)?|discussion|"
                r"interpretation|implications|impact|clinical trial information|trial registration")
PURE_HEADER = re.compile(rf"^\s*(?:{HEADER_WORDS})\s*[:.\-–—]?\s*$", re.I)
LEAD_HEADER = re.compile(rf"^\s*(?:{HEADER_WORDS})\s*[:–—]\s*", re.I)
DROP_PARA = re.compile(r"^\s*(references?|keywords?|disclosures?|funding|acknowledge?ments?)\s*[:.]", re.I)
# figure/table stubs ("Figure 1 Figure 1.") and captions: present in 2021 exports, ~absent in 2025 -> drop for parity
FIG_STUB = re.compile(r"^\s*(figure|fig\.|table)\s*\d+", re.I)
US_STATES = set("""AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH
NJ NM NY NC ND OH OK OR PA PR RI SC SD TN TX UT VT VA WA WV WI WY""".split())
US_NAMES = {"usa", "u.s.a", "us", "u.s", "united states", "united states of america"}
NCT = re.compile(r"\bNCT\s?\d{8}\b", re.I)
INDUSTRY = re.compile(r"\b(Inc\.?|LLC|Ltd\.?|GmbH|Corp(oration)?\.?|Pharma\w*|Therapeutics|Biosciences?|"
                      r"Biotech\w*|Oncology Inc|S\.?A\.?S?\.?|B\.?V\.?|AG|K\.?K\.?|Co\.,)(,|\s|$)")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[\s  -​  　]+")


def unescape(s):
    for _ in range(4):  # Crossref text is often double-escaped (&amp;lt;)
        t = html.unescape(s)
        if t == s:
            break
        s = t
    return s


def body_text(jats):
    # drop whole sections we never want (disclosures, refs, funding ...)
    def sec_filter(m):
        title = re.search(r"<jats:title>(.*?)</jats:title>", m.group(0), re.S)
        return "" if title and DROP_SEC.search(title.group(1)) else m.group(0)
    s = re.sub(r"<jats:sec\b[^>]*>(?:(?!<jats:sec\b).)*?</jats:sec>", sec_filter, jats, flags=re.S)
    s = re.sub(r"<jats:title>.*?</jats:title>", "\n", s, flags=re.S)  # section headers
    paras = re.findall(r"<jats:p\b[^>]*>(.*?)</jats:p>", s, flags=re.S) or [s]
    out = []
    for p in paras:
        p = WS.sub(" ", unescape(TAG.sub(" ", p))).strip()
        if not p or PURE_HEADER.match(p) or DROP_PARA.match(p) or FIG_STUB.match(p):
            continue
        p = LEAD_HEADER.sub("", p, count=1).strip()
        if p:
            out.append(p)
    return "\n\n".join(out)


def country(affil):
    if not affil:
        return None
    last = re.sub(r"^\d+", "", affil.split(",")[-1].strip().rstrip(".")).strip()
    if last in US_STATES or last.lower() in US_NAMES:
        return "United States"
    return last or None


def clean_record(r, year):
    text = body_text(r.get("abstract") or "")
    authors = r.get("author") or []
    affils = [a.get("name", "") for au in authors for a in (au.get("affiliation") or [])]
    first = next((au for au in authors if au.get("sequence") == "first"), authors[0] if authors else {})
    first_aff = ((first.get("affiliation") or [{}])[0]).get("name") if first else None
    page = (r.get("page") or "").split("-")[0]
    n_words = len(text.split())
    return {
        "id": r["DOI"],
        "year": year,
        "abstract_number": page,
        "is_lba": page.upper().startswith("LBA"),
        "title": unescape(TAG.sub("", (r.get("title") or [""])[0])).strip(),
        "n_authors": len(authors),
        "first_author_affiliation": first_aff,
        "first_author_country": country(first_aff),
        "industry_author": any(INDUSTRY.search(a) for a in affils),
        "has_nct": bool(NCT.search(text)),
        "n_words": n_words,
        "eligible": n_words >= MIN_WORDS and not page.upper().startswith("LBA"),
        "text": text,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for y in sys.argv[1:]:
        year = int(y)
        src = os.path.join(ROOT, "data", "raw", f"crossref_{year}.jsonl")
        dest = os.path.join(ROOT, "data", "clean", f"abstracts_{year}.jsonl")
        rows = [clean_record(json.loads(l), year) for l in open(src, encoding="utf-8")]
        tmp = dest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, dest)
        el = [r for r in rows if r["eligible"]]
        words = sorted(r["n_words"] for r in el)
        print(f"{year}: {len(rows)} records, {len(el)} eligible "
              f"(excluded: {sum(r['is_lba'] for r in rows)} LBA, {sum(r['n_words'] < MIN_WORDS for r in rows)} <{MIN_WORDS} words) | "
              f"words median {words[len(words)//2]} IQR {words[len(words)//4]}-{words[3*len(words)//4]} total {sum(words):,} | "
              f"NCT {sum(r['has_nct'] for r in el)} industry {sum(r['industry_author'] for r in el)} "
              f"country known {sum(bool(r['first_author_country']) for r in el)}")


if __name__ == "__main__":
    main()
