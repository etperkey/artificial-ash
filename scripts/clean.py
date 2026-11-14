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
DROP_PARA = re.compile(r"^\s*(references?|keywords?|disclosures?|funding|acknowledge?ments?|title|authors?|"
                       r"affiliations?)\s*[:.]", re.I)
# Headings that the rules above miss (fixed 2026-11-14: 18.8% of 2025 abstracts had one, vs 1-2% in
# other years, because the 2025 export often runs the heading into the first sentence):
HWORD = re.compile(rf"\b(?:{HEADER_WORDS})\b", re.I)
# stray punctuation in front of a heading: ".Methods: Eligible ...", "- Methods: ..."
PUNCT_HEADER = re.compile(rf"^[\s.,;:·•\-–—]+(?=(?:{HEADER_WORDS})\b)", re.I)
# an all-capitals label run into the text: "AIMS AND OBJECTIVES To predict ...", "PURPOSE OF THE STUDY: to ..."
CAPS_LABEL = re.compile(r"^\s*([A-Z][A-Z/&\-]+(?:\s+[A-Z][A-Z/&\-]*)*)\s*(?:[:.]\s+|\s+(?=[A-Z][a-z]))")
# a heading run straight into the first sentence: "Background Acute pain ...", "Results Of 67 pts ..."
RUNIN_HEADER = re.compile(rf"^\s*(?i:{HEADER_WORDS})\s+(?=[A-Z])")
# a heading that lost its paragraph break: "... CAR-T therapy. Methods: Using ..."
MID_HEADER = re.compile(rf"(?<=[.!?)\]])\s+(?i:{HEADER_WORDS})\s*:\s+(?=\S)")


SMALL_WORDS = {"and", "of", "the", "for", "in", "with", "to", "on", "&", "/"}


def label_line(p):
    """A short line that is only a label: "AIM OF THE STUDY AND METHODS", "Impact Statement:",
    "Material and methods:". Not a short sentence such as "Three themes were identified:"."""
    words = p.split()
    if len(words) > 6:
        return False
    if re.search(r"[A-Z]", p) and not re.search(r"[a-z0-9]", p):
        return True
    if not p.rstrip().endswith(":"):
        return False
    return all(w[:1].isupper() or w.strip(":").lower() in SMALL_WORDS or HWORD.fullmatch(w.strip(":"))
               for w in words)
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
        p = PUNCT_HEADER.sub("", p, count=1)
        if not p or PURE_HEADER.match(p) or DROP_PARA.match(p) or FIG_STUB.match(p) or label_line(p):
            continue
        p = LEAD_HEADER.sub("", p, count=1).strip()
        m = CAPS_LABEL.match(p)
        if m and HWORD.search(m.group(1)):
            p = p[m.end():]
        p = RUNIN_HEADER.sub("", p, count=1)
        for q in MID_HEADER.split(p):  # a heading inside a paragraph starts a new one
            q = q.strip()
            if q:
                out.append(q)
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
