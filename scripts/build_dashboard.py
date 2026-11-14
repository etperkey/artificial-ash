"""Build the PUBLIC dashboard page: yearly aggregates plus de-identified passages.

What leaves this script (decided by Eric 2026-11-14):
  * yearly counts, intervals and test results;
  * the passages Pangram highlighted (AI-generated / AI-assisted windows), grouped
    per abstract but WITHOUT title, authors, DOI, abstract number or country, in a
    seeded shuffle (not abstract-number order), with NCT registration numbers masked.
    Unhighlighted text is replaced by a word count;
  * words/phrases significant on two Fisher tests with BH correction (see build_terms).
Nothing else per-abstract goes public. The full viewer (titles, DOIs, whole text)
stays behind the /review/ gate: build_viewer.py.

Usage:  python -I build_dashboard.py
Output: public_dashboard/index.html (atomic write)
"""
import datetime as dt
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASELINE_YEARS = (2020, 2021)     # pooled pre-ChatGPT baseline (pilot is exploratory; the 2026 census will be pre-registered)
FIRST_YEAR = 2020
FULL_N = 250                      # planned per-year pilot sample
NO_TEXT = {2022: "No abstract text in Crossref"}
PENDING = {2026: "All abstracts, after Blood posts them (about Dec 11)"}

# Passages
SHUFFLE_SEED = 20261114           # page order must not follow abstract numbers
WIN_CODE = {"AI-Generated": "g", "AI-Assisted": "a"}
NCT_RE = re.compile(r"NCT\s?\d{6,8}", re.I)
# A few abstracts carry their title (or author/affiliation lines) pasted into the body;
# the detector scored them as text, so they are masked for display rather than cleaned away.
HEADER_LINE_RE = re.compile(r"(?im)^[ \t]*(title|authors?|affiliations?)[ \t]*:[^\n]*")

# Words and phrases (1-3 word n-grams). Two two-sided Fisher exact tests per term, each
# with Benjamini-Hochberg FDR across every term in its family; a term is listed only when
# BOTH are significant in the "more" direction (Eric chose Fisher + BH, 2026-11-14).
#   Year test (abstract-level): abstracts containing the term, latest year vs pooled 2020-21.
#   Highlighted test (occurrence-level): occurrences in highlighted vs unhighlighted text of
#     the ERA_FROM+ abstracts. Occurrences cluster within abstracts (a drug name repeated in
#     one flagged abstract reaches q ~ 1e-25), so this test is optimistic on its own; the
#     year test and MIN_ABSTRACTS are what guard against that.
# Families are defined by marginal counts only (>= MIN_TESTED), which Fisher's test
# conditions on, so the filter does not bias the tests.
ERA_FROM = 2023                   # highlighted vs unhighlighted text from these years on
FDR_Q = 0.05
MIN_TESTED = 10                   # family membership: >= 10 abstracts (year) / occurrences (highlighted)
MIN_ABSTRACTS = 6                 # display floor: in >= 6 different highlighted abstracts
MIN_RATIO = 5.0                   # display floor: > 5 times as common in highlighted text (Eric, 2026-11-14)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
WORD_RE = re.compile(r"[a-z]{2,}(?:['\-][a-z]+)*")


def load_raw(year_of):
    """id -> (year, Pangram response) for every successfully scored abstract."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.raw.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            res = x["response"]
            if res.get("stage") != "STAGE_SUCCESS" or x["id"] in out or x["id"] not in year_of:
                continue
            out[x["id"]] = (year_of[x["id"]], res)
    return out


def n_words(s):
    return len(s.split())


def build_passages(raw):
    """Per abstract with any highlighted window: year, label, word counts and an
    ordered list of items. An int item is a run of unhighlighted words (count only);
    a list item is a highlighted block of [code, text, [confidences]] segments."""
    ids = sorted(raw)
    random.Random(SHUFFLE_SEED).shuffle(ids)
    out = []
    for aid in ids:
        year, res = raw[aid]
        text = res["text"]
        wins = sorted(res.get("windows", []), key=lambda w: w["start_index"])
        items, pos, hw = [], 0, 0
        for w in wins:
            s, e, lab = w["start_index"], w["end_index"], w["label"]
            if lab == "Human Written" or e <= s:
                continue
            if lab not in WIN_CODE:
                raise ValueError(f"unknown Pangram window label {lab!r}")
            code, conf = WIN_CODE[lab], (w.get("confidence") or "").lower()
            between, seg = (text[pos:s] if s > pos else ""), text[max(s, pos):e]
            if items and isinstance(items[-1], list) and not between.strip():
                last = items[-1][-1]
                if last[0] == code:
                    last[1] += between + seg
                    last[2].add(conf)
                else:
                    last[1] += between
                    items[-1].append([code, seg, {conf}])
            else:
                if between.strip():
                    items.append(n_words(between))
                items.append([[code, seg, {conf}]])
            hw += n_words(seg)
            pos = e
        if not items:
            continue
        if text[pos:].strip():
            items.append(n_words(text[pos:]))
        for it in items:
            if isinstance(it, list):
                it[0][1] = it[0][1].lstrip()
                it[-1][1] = it[-1][1].rstrip()
                for sg in it:
                    sg[1] = NCT_RE.sub("NCT[…]", sg[1])
                    sg[1] = HEADER_LINE_RE.sub(lambda m: f"[{m.group(1).lower()} removed]", sg[1])
                    sg[2] = sorted(c for c in sg[2] if c)
        nw = n_words(text)
        out.append({"y": year, "lab": res["prediction_short"], "nw": nw, "hw": min(hw, nw), "p": items})
    return out


def grams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            if all(WORD_RE.fullmatch(t) for t in tokens[i:i + n])]


def build_terms(raw):
    """Words (n=1) and phrases (n=2,3) that are significantly more common in highlighted
    than unhighlighted text AND significantly more common in latest-year abstracts than in
    2020-21 abstracts (two Fisher tests, BH within each family). Year shares are of all
    sampled abstracts in a year that contain the term anywhere."""
    n_year = Counter(y for y, _ in raw.values())
    years = sorted(n_year)
    latest = max(years)
    n_base = sum(n_year[y] for y in BASELINE_YEARS)
    toks = {aid: TOKEN_RE.findall(res["text"].lower()) for aid, (_, res) in raw.items()}
    in_year = defaultdict(Counter)            # year -> term -> abstracts containing it
    hl, hu = Counter(), Counter()             # term -> occurrences (ERA_FROM+ text)
    nhl, nhu = Counter(), Counter()           # n -> total n-gram slots, highlighted / not
    hl_abs = defaultdict(set)
    for n in (1, 2, 3):
        for aid, (y, res) in raw.items():
            for g in set(grams(toks[aid], n)):
                in_year[y][g] += 1
            if y < ERA_FROM:
                continue
            for w in res.get("windows", []):
                gs = grams(TOKEN_RE.findall(res["text"][w["start_index"]:w["end_index"]].lower()), n)
                if w["label"] == "Human Written":
                    hu.update(gs); nhu[n] += len(gs)
                else:
                    hl.update(gs); nhl[n] += len(gs)
                    for g in gs:
                        hl_abs[g].add(aid)
    size = lambda g: g.count(" ") + 1
    base = lambda g: sum(in_year[y][g] for y in BASELINE_YEARS)

    # year test: latest vs pooled baseline, abstract-level
    fam_a = sorted(g for g in set(in_year[latest]).union(*(in_year[y] for y in BASELINE_YEARS))
                   if in_year[latest][g] + base(g) >= MIN_TESTED)
    p_a = [fisher_exact([[in_year[latest][g], n_year[latest] - in_year[latest][g]],
                         [base(g), n_base - base(g)]])[1] for g in fam_a]
    q_a = dict(zip(fam_a, multipletests(p_a, method="fdr_bh")[1]))
    # highlighted test: highlighted vs unhighlighted ERA_FROM+ text, occurrence-level
    fam_b = sorted(g for g in set(hl) | set(hu) if hl[g] + hu[g] >= MIN_TESTED)
    p_b = [fisher_exact([[hl[g], nhl[size(g)] - hl[g]], [hu[g], nhu[size(g)] - hu[g]]])[1] for g in fam_b]
    q_b = dict(zip(fam_b, multipletests(p_b, method="fdr_bh")[1]))

    found = {}
    for g in set(q_a) & set(q_b):
        rate_hl, rate_hu = hl[g] / nhl[size(g)], hu[g] / nhu[size(g)]
        rose = in_year[latest][g] / n_year[latest] > base(g) / n_base
        ratio = rate_hl / ((hu[g] + 0.5) / nhu[size(g)])
        if (q_a[g] < FDR_Q and q_b[g] < FDR_Q and rose and rate_hl > rate_hu
                and len(hl_abs[g]) >= MIN_ABSTRACTS and ratio > MIN_RATIO):
            found[g] = {"t": g, "n": size(g), "k": len(hl_abs[g]), "h": hl[g], "u": hu[g],
                        "r": round(ratio, 1),
                        "qa": float(q_a[g]), "qb": float(q_b[g]),
                        "b": round(100 * base(g) / n_base, 1),
                        "s": [round(100 * in_year[y][g] / n_year[y], 1) for y in years]}
    # a longer term found in nearly the same highlighted abstracts (>= 80%) as a listed shorter
    # term inside it adds nothing ("underscore the" next to "underscore"): list only the shorter
    for g in sorted(found, key=lambda t: -size(t)):
        if any(o != g and size(o) < size(g) and f" {o} " in f" {g} " and found[g]["k"] >= 0.8 * found[o]["k"]
               for o in found):
            del found[g]
    key = lambda v: (-v["r"], v["t"])
    return {"years": years, "latest": latest, "base_label": "–".join(map(str, BASELINE_YEARS)),
            "words": sorted((v for v in found.values() if v["n"] == 1), key=key),
            "phrases": sorted((v for v in found.values() if v["n"] > 1), key=key),
            "rule": {"q": FDR_Q, "min_tested": MIN_TESTED, "abstracts": MIN_ABSTRACTS, "ratio": MIN_RATIO,
                     "hl_share": round(100 * nhl[1] / (nhl[1] + nhu[1]), 1),
                     "era_from": ERA_FROM, "n_year_tests": len(fam_a), "n_hl_tests": len(fam_b)}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    by_year, year_of = {}, {}
    for p in glob.glob(os.path.join(ROOT, "results", "*.scores.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("stage") == "STAGE_SUCCESS":
                by_year.setdefault(r["year"], {})[r["id"]] = r["prediction_short"]
                year_of[r["id"]] = r["year"]
    raw = load_raw(year_of)
    missing = set(year_of) - set(raw)
    if missing:
        raise SystemExit(f"{len(missing)} scored abstracts have no raw Pangram response")
    bad = [i for i, (_, res) in raw.items() if res["prediction_short"] != by_year[year_of[i]][i]]
    if bad:
        raise SystemExit(f"{len(bad)} abstracts disagree between scores and raw files")

    rows = []
    for y in range(FIRST_YEAR, 2027):
        if y in NO_TEXT:
            rows.append({"year": y, "status": "nodata", "note": NO_TEXT[y]}); continue
        if y in PENDING and y not in by_year:
            rows.append({"year": y, "status": "pending", "note": PENDING[y]}); continue
        if y not in by_year:
            continue
        labs = list(by_year[y].values()); n = len(labs)
        ai, mixed = labs.count("AI"), labs.count("Mixed")
        lo, hi = proportion_confint(ai + mixed, n, method="wilson")
        rows.append({"year": y, "status": "scored", "n": n, "ai": ai, "mixed": mixed,
                     "flag": ai + mixed, "lo": lo, "hi": hi, "partial": n < FULL_N})

    brows = [r for r in rows if r["year"] in BASELINE_YEARS and r["status"] == "scored"]
    bn, bk = sum(r["n"] for r in brows), sum(r["flag"] for r in brows)
    blo, bhi = proportion_confint(bk, bn, method="wilson")
    base = {"label": "–".join(str(r["year"]) for r in brows), "n": bn, "flag": bk, "lo": blo, "hi": bhi}
    for r in brows:
        r["is_base"] = True
    comp = [r for r in rows if r["status"] == "scored" and r["year"] not in BASELINE_YEARS]
    ps = [fisher_exact([[r["flag"], r["n"] - r["flag"]], [bk, bn - bk]])[1] for r in comp]
    if ps:
        for r, q in zip(comp, multipletests(ps, method="fdr_bh")[1]):
            r["q"] = float(q)
    passages = build_passages(raw)
    for a in passages:  # refuse to publish a pasted title, author or affiliation line
        for it in a["p"]:
            for sg in (it if isinstance(it, list) else []):
                if HEADER_LINE_RE.search(sg[1]):
                    raise SystemExit(f"a title/author/affiliation line survived masking: {sg[1][:60]!r}")
    terms = build_terms(raw)
    # ASH_BUILD_DATE=YYYY-MM-DD pins the "Updated" stamp so a rebuild can be compared byte for byte
    day = dt.date.fromisoformat(os.environ["ASH_BUILD_DATE"]) if os.environ.get("ASH_BUILD_DATE") else dt.date.today()
    updated = f"{day.day} {day.strftime('%b %Y')}"
    hl_by_year = Counter(p["y"] for p in passages)
    data = {"updated": updated, "model": "Pangram 4.1", "rows": rows, "base": base, "n_tests": len(ps),
            "hl": {str(r["year"]): hl_by_year.get(r["year"], 0) for r in rows if r["status"] == "scored"},
            "hl_lab": dict(sorted(Counter(p["lab"] for p in passages).items())),
            "terms": terms}
    pas_data = {"updated": updated, "model": "Pangram 4.1",
                "rows": [{"year": r["year"], "n": r["n"]} for r in rows if r["status"] == "scored"],
                "passages": passages}
    # Fisher p-values differ in the 16th significant digit between operating systems' math
    # libraries; round computed statistics so the page is byte-identical everywhere
    # (the page shows 2-3 digits; which terms are listed was decided above, unrounded).
    for r in rows + [base]:
        for k in ("lo", "hi", "q"):
            if isinstance(r.get(k), float):
                r[k] = float(f"{r[k]:.8g}")
    for t in terms["words"] + terms["phrases"]:
        for k in ("qa", "qb"):
            t[k] = float(f"{t[k]:.8g}")
    out_dir = os.path.join(ROOT, "public_dashboard")
    pages = [("index.html", MAIN_TITLE, MAIN_DESC, MAIN_BODY, data),
             (os.path.join("passages", "index.html"), PAS_TITLE, PAS_DESC, PAS_BODY, pas_data)]
    for rel, title, desc, body, d in pages:
        payload = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        # no identifiers may ride along with the passages
        for leak in ("10.1182/", "blood-20"):
            if leak in payload:
                raise SystemExit(f"identifier pattern {leak!r} found in the public payload")
        html = (HEAD.replace("__TITLE__", title).replace("__DESC__", desc)
                + body.replace("__DATA__", payload.replace("</", "<\\/")))
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".tmp", "w", encoding="utf-8", newline="\n") as f:  # same bytes on every OS
            f.write(html)
        os.replace(dest + ".tmp", dest)
    for r in rows:
        print(r)
    print(f"passages: {len(passages)} abstracts with highlighted text; "
          f"terms: {len(terms['words'])} words, {len(terms['phrases'])} phrases")
    for rel, *_ in pages:
        dest = os.path.join(out_dir, rel)
        print("->", dest, f"({os.path.getsize(dest) / 1e3:.0f} kB)")


MAIN_TITLE = "AI Text in ASH Abstracts"
MAIN_DESC = ("How often an AI-text detector flags American Society of Hematology annual-meeting abstracts, "
             "by year, with every passage it highlighted and the words those passages lean on. Preliminary "
             "pilot results by Eric Perkey, Frank Wen, Reid Shaw and Max Wen.")
PAS_TITLE = "Highlighted ASH Passages"
PAS_DESC = ("Every passage an AI-text detector highlighted in random samples of American Society of "
            "Hematology annual-meeting abstracts, 2020-2025, shown without titles or authors.")

HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<meta name="description" content="__DESC__">
<style>
:root{
  --bg:#fbfaf7; --panel:#ffffff; --ink:#1a1a1a; --ink2:#52514e; --mute:#7a786f; --rule:#e5e2db;
  --grid:#ecebe6; --axis:#c3c2b7; --link:#2a4a7a; --chip:#efece6; --note:#f6efe2; --note-ink:#6b4a12;
  --s1:#2a78d6; --s2:#eb6834; --ring:rgba(11,11,11,.10);
  --hl:rgba(236,178,20,.34); --hl-soft:rgba(236,178,20,.13); --hl-edge:#a87a00; --find:#2a4a7a;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#15161a; --panel:#1c1e23; --ink:#e8e6e1; --ink2:#c3c2b7; --mute:#9a948a; --rule:#2a2d33;
    --grid:#2a2c31; --axis:#3b3d43; --link:#8ab0e0; --chip:#262830; --note:#2a2419; --note-ink:#e7c78f;
    --s1:#3987e5; --s2:#d95926; --ring:rgba(255,255,255,.10);
    --hl:rgba(240,186,60,.27); --hl-soft:rgba(240,186,60,.10); --hl-edge:#e0b04a; --find:#8ab0e0;
  }
}
:root[data-theme="dark"]{
  --bg:#15161a; --panel:#1c1e23; --ink:#e8e6e1; --ink2:#c3c2b7; --mute:#9a948a; --rule:#2a2d33;
  --grid:#2a2c31; --axis:#3b3d43; --link:#8ab0e0; --chip:#262830; --note:#2a2419; --note-ink:#e7c78f;
  --s1:#3987e5; --s2:#d95926; --ring:rgba(255,255,255,.10);
  --hl:rgba(240,186,60,.27); --hl-soft:rgba(240,186,60,.10); --hl-edge:#e0b04a; --find:#8ab0e0;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:17px/1.6 ui-serif,Georgia,"Iowan Old Style",Charter,serif;-webkit-font-smoothing:antialiased}
.sans,.ui,table,.hero,.chip,svg text,.tip,summary,.legend{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link)}
.wrap{max-width:860px;margin:0 auto;padding:max(28px,env(safe-area-inset-top,0px)) max(16px,env(safe-area-inset-right,0px)) max(72px,calc(env(safe-area-inset-bottom,0px) + 56px)) max(16px,env(safe-area-inset-left,0px))}
.back{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--mute);text-decoration:none}
.back:hover{text-decoration:underline}
h1{font-size:30px;line-height:1.2;margin:14px 0 6px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 12px;font-size:17px}
.byline{font:14.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);margin:0 0 12px}
.byline sup{font-size:.7em;color:var(--mute)} .byline span{color:var(--mute);font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 22px}
.chip{font-size:12.5px;background:var(--chip);border:1px solid var(--rule);border-radius:999px;padding:3px 10px;color:var(--ink2)}
.chip.warn{background:var(--note);color:var(--note-ink);border-color:transparent}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:20px 20px 16px;margin:0 0 22px}
.hero{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px}
.hero .big{font-size:56px;font-weight:650;line-height:1;letter-spacing:-.02em}
.hero .lede{font-size:16px;color:var(--ink2);max-width:520px;line-height:1.45}
.hero .vs{flex-basis:100%;font-size:14px;color:var(--mute);margin-top:6px}
h2{font-size:20px;margin:0 0 4px}
.cap{color:var(--mute);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 10px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13.5px;color:var(--ink2);margin:6px 0 4px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
.legend .whisk{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:6px;
  background:linear-gradient(var(--ink2),var(--ink2)) center/1.5px 100% no-repeat,linear-gradient(var(--ink2),var(--ink2)) center top/8px 1.5px no-repeat,linear-gradient(var(--ink2),var(--ink2)) center bottom/8px 1.5px no-repeat}
#chart{position:relative;width:100%}
#chart svg{display:block;width:100%;height:auto;overflow:visible}
.hit{fill:transparent;cursor:default;outline:none}
.hit:focus-visible{stroke:var(--link);stroke-width:1.5}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--rule);box-shadow:0 4px 18px rgba(0,0,0,.12);
  border-radius:8px;padding:8px 10px;font-size:13px;line-height:1.45;min-width:180px;opacity:0;transition:opacity .08s;z-index:3}
.tip .h{font-weight:650;margin-bottom:3px}
.tip .row{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tip .row span:first-child{display:flex;align-items:center;gap:6px;color:var(--ink2)}
.tip .key{display:inline-block;width:12px;height:2.5px;border-radius:2px}
.tip b{font-variant-numeric:tabular-nums}
.tip .m{color:var(--mute);font-size:12px;margin-top:3px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--link);font-size:14px}
.tablewrap{overflow-x:auto;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:600;font-size:12.5px}
.prose h2{margin-top:6px}
.prose p,.prose li{font-size:16.5px}
.prose ul{padding-left:20px;margin:6px 0 12px}
.fine{color:var(--mute);font:13.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin-top:28px;border-top:1px solid var(--rule);padding-top:14px}
.hit.go{cursor:pointer}
.golink{display:inline-block;font:600 15px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--link);margin:4px 0 2px}
[hidden]{display:none!important}
/* words and phrases */
.tabs{display:flex;gap:4px;margin:8px 0 2px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.tabs button{font:inherit;font-size:14px;padding:5px 12px;border:1px solid var(--rule);border-radius:999px;background:transparent;color:var(--ink2);cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--chip);color:var(--ink);font-weight:600}
table.terms td,table.terms th{white-space:normal;vertical-align:middle}
.termbtn{font:inherit;font-weight:600;font-size:14.5px;color:var(--link);background:none;border:0;padding:0;cursor:pointer;text-align:left;text-decoration:none}
.termbtn:hover{text-decoration:underline}
.sub2{display:block;color:var(--mute);font-size:12.5px;font-weight:400}
.spark{display:flex;align-items:center;justify-content:flex-end;gap:8px}
.spark svg{flex:none;display:block;overflow:visible}
.spark span{min-width:96px;text-align:right;white-space:nowrap}
.linkbtn{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--link);background:none;border:0;padding:6px 0;cursor:pointer}
.linkbtn:hover{text-decoration:underline}
/* passage explorer */
.controls{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin:10px 0 6px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.controls select,.controls input{font:inherit;font-size:14px;padding:6px 8px;border:1px solid var(--rule);border-radius:8px;background:var(--bg);color:var(--ink);max-width:100%}
.controls input[type=search]{flex:1 1 200px;min-width:0}
.legend mark{padding:0 4px}
.count{font:14px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink2);margin:10px 0 4px}
.count button{margin-left:6px}
.ab{border-top:1px solid var(--rule);padding:16px 0 6px}
.ab-h{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;font:13.5px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--mute);margin-bottom:8px}
.ab-h b{color:var(--ink);font-weight:650}
.ab-h .lab{display:inline-flex;align-items:center;gap:6px;color:var(--ink2)}
.ab-h .lab i{display:inline-block;width:10px;height:10px;border-radius:3px}
.ptext{white-space:pre-wrap;margin:0;font-size:16px;line-height:1.65}
.pc{font:12.5px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--mute);margin:3px 0 0}
.gap{display:flex;align-items:center;gap:10px;font:12.5px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--mute);margin:10px 0}
.gap::before,.gap::after{content:"";flex:1;border-top:1px dashed var(--axis)}
mark{color:inherit;border-radius:2px;padding:1px 0;-webkit-box-decoration-break:clone;box-decoration-break:clone}
mark.g{background:var(--hl);box-shadow:inset 0 -1.5px 0 var(--hl-edge)}
mark.a{background:var(--hl-soft);text-decoration:underline dotted var(--hl-edge);text-decoration-thickness:1.5px;text-underline-offset:3px}
.find{outline:1.5px solid var(--find);outline-offset:1px;border-radius:2px}
.empty{font:15px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink2);padding:18px 0}
.pager{display:flex;align-items:center;justify-content:center;gap:12px;margin-top:14px;font:14px system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--mute)}
.pager button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);cursor:pointer}
.pager button:disabled{opacity:.4;cursor:default}
@media (forced-colors:active){mark.g{text-decoration:underline;text-decoration-thickness:2px}.find{outline-color:Highlight}}
@media (max-width:768px){h1{font-size:26px}.hero .big{font-size:52px}.card{padding:16px 14px 12px}}
@media (max-width:520px){table.terms th,table.terms td{padding:7px 4px}.termbtn{font-size:14px}.spark span{min-width:0}.spark{flex-direction:column;align-items:flex-end;gap:2px}.spark svg{width:60px;height:auto}.ptext{font-size:15.5px}}
@media (max-width:430px){h1{font-size:24px}.hero .big{font-size:48px}body{font-size:16px}.prose p,.prose li{font-size:16px}}
@media (max-width:360px){h1{font-size:22px}.spark svg{display:none}.nabs{display:none}table.terms th,table.terms td{padding:7px 3px}}
</style>
</head>
"""

MAIN_BODY = r"""<body>
<div class="wrap">
<a class="back" href="/">&larr; eperkey.org</a>
<h1>How much ASH abstract text is AI-written?</h1>
<p class="sub">An AI-text detector run on random samples of American Society of Hematology annual-meeting abstracts, before and after ChatGPT.</p>
<p class="byline">Eric Perkey<sup>1*</sup>, Frank Wen<sup>1,2*</sup>, Reid Shaw<sup>1*</sup>, Max Wen<sup>2</sup><br>
<span><sup>1</sup>Section of Hematology/Oncology, University of Chicago &middot; <sup>2</sup>Wen Family Institute &middot; <sup>*</sup>Contributed equally</span></p>
<div class="chips"><span class="chip warn">Preliminary pilot &middot; not peer reviewed</span><span class="chip" id="upd"></span><span class="chip" id="mdl"></span></div>

<section class="card" aria-labelledby="hero-h">
  <h2 id="hero-h" class="sans" style="font-size:13px;font-weight:600;color:var(--mute);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px">Headline</h2>
  <div class="hero" id="hero"></div>
</section>

<section class="card" aria-labelledby="ch-h">
  <h2 id="ch-h">Share of abstracts flagged, by meeting year</h2>
  <p class="cap">Each column is one year's random sample. Whiskers are 95% confidence intervals for the total flagged. Hover or tab to a year for the counts; click or press Enter to read that year's highlighted passages.</p>
  <div class="legend" aria-hidden="true"><span><i style="background:var(--s1)"></i>AI-written</span><span><i style="background:var(--s2)"></i>Partly AI-written or AI-edited</span><span><span class="whisk"></span>95% interval</span></div>
  <div id="chart" role="img" aria-describedby="ch-sum"></div>
  <p id="ch-sum" class="cap" style="margin-top:6px"></p>
  <details><summary>Show the data table</summary><div class="tablewrap"><table id="tbl"></table></div></details>
</section>

<section class="card" aria-labelledby="w-h">
  <h2 id="w-h">Words and phrases the highlighted passages lean on</h2>
  <p class="cap" id="w-rule"></p>
  <div class="tabs" role="tablist" aria-label="Words or phrases">
    <button role="tab" id="tab-words" aria-selected="true" aria-controls="wt">Words</button>
    <button role="tab" id="tab-phrases" aria-selected="false" aria-controls="wt">Phrases</button>
  </div>
  <div class="tablewrap"><table class="terms" id="wt" role="tabpanel"></table></div>
  <button class="linkbtn" id="wt-more"></button>
</section>

<section class="card" aria-labelledby="ex-h">
  <h2 id="ex-h">Read the highlighted passages</h2>
  <p class="cap" id="ex-intro"></p>
  <a class="golink" href="passages/">Open the passage explorer &rarr;</a>
</section>

<section class="card prose" aria-labelledby="m-h">
  <h2 id="m-h">What was done</h2>
  <ul>
    <li><b>Abstracts.</b> ASH annual-meeting abstracts as published in the <i>Blood</i> meeting supplements, retrieved from Crossref. Only the abstract body was analyzed: titles, author lists, disclosures, references and section labels were stripped with the same rules for every year. 2022 is missing because Crossref does not carry that year's abstract text.</li>
    <li><b>Sampling.</b> A simple random sample from each year (fixed random seed) of abstracts with at least 75 words, leaving out late-breaking abstracts: <span id="sampling"></span>.</li>
    <li><b>Detector.</b> <span id="mdl2"></span> (Pangram Labs), a commercial AI-text detector. It labels each abstract as human-written, mixed (some passages AI-written or AI-edited) or AI-written. "Flagged" here means mixed or AI-written.</li>
    <li><b>Highlighted passages.</b> The detector also judges each stretch of an abstract (from a sentence to several paragraphs) on its own and labels it human-written, AI-generated or AI-assisted. The highlights are the stretches given one of the last two labels. <span id="hl-human"></span></li>
    <li><b>Words and phrases.</b> Every single word and run of two or three words was tested twice with Fisher's exact test (two-sided). The first test, for terms used at least 10 times, compares how often a term occurs in the highlighted versus the unhighlighted text of the <span class="era"></span> abstracts. The second, for terms found in at least 10 abstracts, compares how many abstracts contain the term in the latest year versus 2020&ndash;2021. Each test's p-values are adjusted across all terms tested (Benjamini&ndash;Hochberg). A term is listed only if it passes both tests, appears in at least 6 different highlighted abstracts and is more than 5 times as common in highlighted as in unhighlighted text; those two cut-offs were applied after the tests, to keep the list short. &ldquo;Times as common&rdquo; compares the term's rate per word in the two kinds of text, with 0.5 added to the unhighlighted count so that rare terms are not overstated. When a phrase turns up in nearly the same abstracts as a shorter term inside it, only the shorter term is listed. The list is exploratory. The detector may highlight a passage partly because it contains words like these, so the list describes highlighted passages; it does not show which words AI writes. Repeats of a word within one abstract make the first test optimistic, and most highlighted abstracts are from 2025, so changes in writing style over the years play a part too. With 250 abstracts a year, rarer style words may not reach significance.</li>
    <li><b>Baseline.</b> ASH 2020 and 2021 abstracts were submitted in August of those years, more than a year before ChatGPT launched on 30 November 2022, though earlier tools such as GPT-3, Grammarly and DeepL already existed. <span id="basetxt"></span></li>
    <li><b>Statistics.</b> Wilson 95% intervals for each share. Each later year is compared with 2020 and 2021 combined by Fisher's exact test; p-values are adjusted for the number of comparisons (Benjamini&ndash;Hochberg).</li>
  </ul>
  <h2>How to read it</h2>
  <ul>
    <li>A flag is a detector's judgment, not proof. It says nothing about whether the science is sound or whether AI use broke any rule; using AI to polish English is common and usually allowed.</li>
    <li>How well the detector catches AI text in short, number-heavy clinical abstracts has not yet been measured here. Tests on abstracts deliberately written or edited by AI are planned.</li>
    <li>With 250 abstracts a year, each share is uncertain by several percentage points either way; the whiskers show how much.</li>
    <li>Highlighted passages are shown without titles, authors, abstract numbers or links, and not in abstract-number order. Trial registration numbers and pasted title lines are masked, and text the detector did not highlight is reduced to a word count. Passages can still name an institution, a study or a data set. Full per-abstract results are not published.</li>
  </ul>
  <h2>What comes next</h2>
  <p>Every ASH 2026 abstract (about 8,000) will be scored once <i>Blood</i> posts them, under an analysis plan registered in advance. Planned comparisons include abstracts by fully academic author teams, by teams entirely from pharmaceutical or other companies, and by mixed academic and industry teams, as well as comparisons by scientific category, presentation type and region, all with false-discovery-rate control.</p>
  <h2>Related work</h2>
  <ul>
    <li>Howard FM, et al. Characterizing the increase in artificial intelligence content detection in oncology scientific abstracts from 2021 to 2023. <i>JCO Clin Cancer Inform.</i> 2024;8:e2400077. <a href="https://doi.org/10.1200/CCI.24.00077">doi:10.1200/CCI.24.00077</a></li>
    <li>Kobak D, et al. Delving into LLM-assisted writing in biomedical publications through excess vocabulary. <i>Sci Adv.</i> 2025;11(27):eadt3813. <a href="https://doi.org/10.1126/sciadv.adt3813">doi:10.1126/sciadv.adt3813</a></li>
  </ul>
</section>

<section class="card prose" aria-labelledby="s-h">
  <h2 id="s-h">What could work better than a ban</h2>
  <p>ASH's rules for meeting abstracts already forbid AI-generated text. Its <a href="https://www.hematology.org/meetings/annual-meeting/abstracts/call-for-abstracts">2026 call for abstracts</a> makes an abstract ineligible if AI was used to generate its text, and its <a href="https://www.hematology.org/meetings/annual-meeting/abstracts/copyright-and-reuse-policy">copyright policy</a> bars authors from putting their abstract into any AI tool for any purpose, translation included, whatever the tool's privacy settings. The call does allow spelling and grammar checks with a &ldquo;closed AI system&rdquo;, and using AI to revise existing text with the authors' oversight. Those lines are hard to draw now that most writing software has AI built in. The detector labeled <span id="ai-last"></span> AI-written. Some of that may be heavy AI revision, which ASH allows, rather than AI-generated text, which it does not, and a detector cannot tell the two apart. So the rule cannot be enforced fairly with a detector, authors cannot easily tell where the line is, and a ban pushes AI use out of sight. Some alternatives:</p>
  <ul>
    <li><b>Disclose, don't ban.</b> Add an AI section to the disclosure form authors already fill in for conflicts of interest: which tool and version, what it did (language editing, translation, drafting text, literature search, code, statistical analysis, figures or data handling) and what the authors checked. Publish it with the abstract, as conflicts of interest are. Disclosed use should not count against an abstract; undisclosed AI writing should. The <a href="https://www.icmje.org/recommendations/browse/artificial-intelligence/ai-use-by-authors.html">ICMJE recommends</a> the same for medical journals: disclosure at submission, and a description of the use in the work itself.</li>
    <li><b>Marking at the source.</b> The European Union already requires it. Article 50(2) of the <a href="https://eur-lex.europa.eu/eli/reg/2024/1689/oj">EU AI Act</a> has applied since 2 August 2026: providers of generative AI, large language models included, must mark what their systems produce in a machine-readable way, such as a watermark, so that it can be detected as AI-generated, as far as is technically feasible. Systems already on the market have until 2 December 2026. The Commission's <a href="https://digital-strategy.ec.europa.eu/en/news/commission-publishes-code-practice-marking-and-labelling-ai-generated-content">Code of Practice</a> (June 2026) sets out how. China has required visible labels and embedded marks on AI-generated content since September 2025. Marks have limits: rewording or translation can weaken a text watermark, a model run on a researcher's own computer may add none, and the EU rule exempts AI that only assists with standard editing, the same line ASH draws between polishing and writing. Marks can help check a disclosure; they cannot replace one.</li>
    <li><b>Reproducible computation.</b> When AI helps handle or analyze data, someone else should be able to re-run the analysis: the code, a pinned software environment, random seeds, the exact model name and version, the prompts and the saved model outputs. Saved outputs matter because hosted models change and are retired, so a re-run months later may not give the same answer. Code written by an AI model is still code, and it should be shared and run like any other.</li>
    <li><b>Free to run, not a subscription.</b> Some journals now offer runnable &ldquo;compute capsules&rdquo;. Several Nature Portfolio journals have <a href="https://blogs.nature.com/blog/nature-research-journals-trial-new-tools-to-enhance-code-peer-review-and-publication/">reviewed code through Code Ocean</a> since 2018, a commercial platform that sells institutional subscriptions and runs capsules on its own servers. The idea is right, but whether a paper can be checked should not depend on one company's servers and prices, or on a reader's institution paying for access. Free options already exist. <a href="https://zenodo.org/">Zenodo</a> (run by CERN) and <a href="https://www.softwareheritage.org/">Software Heritage</a> store code permanently, a Docker file records the exact software versions used, and <a href="https://mybinder.org/">mybinder.org</a> lets anyone rerun an analysis in a web browser at no charge. Societies, journals and funders should cover the cost of keeping these free.</li>
    <li><b>Detectors for monitoring, not for judging individuals.</b> Detectors can misjudge some writers: in <a href="https://doi.org/10.1016/j.patter.2023.100779">one study</a>, popular detectors labeled more than half of essays by non-native English writers as AI-written. Pangram flagged none of the 500 pre-ChatGPT abstracts here, but that does not show it treats every group of writers alike. Detectors suit what this page does, tracking a field in aggregate, not rejecting a single abstract. AI language editing is also a real help to the many ASH authors who write in a second language, and a ban falls hardest on them.</li>
    <li><b>Doing the same here.</b> We plan to release this audit as a free, re-runnable package: the code, a pinned environment and a check that freshly downloaded abstract text matches published fingerprints. The abstracts stay with ASH: they are downloaded again from Crossref rather than redistributed. The detector step is the one part that can only be repeated with a paid Pangram key.</li>
  </ul>
</section>

<p class="fine">Independent analysis by Eric Perkey, Frank Wen, Reid Shaw and Max Wen; views are the authors' own. Not affiliated with or endorsed by the American Society of Hematology or Pangram Labs. Abstract text &copy; the American Society of Hematology; passages the detector highlighted are quoted for research and commentary.</p>
<p class="fine">AI use: the language on this page was generated by Claude (Anthropic) at the authors' direction, and Claude helped write the analysis code. Pangram scored the abstracts. The authors are responsible for the content.</p>
</div>

<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const pct = (k, n, d = 1) => (100 * k / n).toFixed(d) + '%';
const fmtP = q => q < 0.001 ? '< 0.001' : q.toFixed(3);
$('upd').textContent = 'Updated ' + DATA.updated;
$('mdl').textContent = DATA.model;
$('mdl2').textContent = DATA.model;
const rows = DATA.rows, scored = rows.filter(r => r.status === 'scored');
const base = DATA.base, last = scored[scored.length - 1];
$('sampling').textContent = scored.every(r => r.n === scored[0].n)
  ? `${scored[0].n} abstracts from each of ${scored.slice(0, -1).map(r => r.year).join(', ')} and ${last.year}`
  : scored.map(r => `${r.n} from ${r.year}${r.partial ? ' (partial)' : ''}`).join(', ');
$('ai-last').textContent = `${pct(last.ai, last.n, 0)} of the ${last.n} sampled ${last.year} abstracts`;
$('basetxt').textContent = `The detector flagged ${base.flag} of these ${base.n} abstracts (95% upper bound ${(100 * base.hi).toFixed(1)}%). That bounds its false alarms on writing from ${base.label} only; it does not measure its error rate on current abstracts.`;
$('hl-human').textContent = `Of the ${scored.reduce((s, r) => s + DATA.hl[r.year], 0)} abstracts with any highlight, ${DATA.hl_lab.Human || 0} were labeled human-written overall; they are included.`;

// hero
(function(){
  const h = $('hero');
  const big = document.createElement('div'); big.className = 'big'; big.textContent = pct(last.flag, last.n, 0);
  const lede = document.createElement('div'); lede.className = 'lede';
  lede.textContent = `of a random sample of ${last.n} ASH ${last.year} abstracts were flagged by the detector as AI-written in whole or in part (${pct(last.ai, last.n, 0)} labeled AI-written overall).`;
  const vs = document.createElement('div'); vs.className = 'vs';
  vs.textContent = `For comparison: ${pct(base.flag, base.n, 0)} of ${base.n} ASH ${base.label} abstracts, written before ChatGPT was released.`;
  h.append(big, lede, vs);
})();

// table
(function(){
  const t = $('tbl'); const hd = t.createTHead().insertRow();
  ['Year', 'Abstracts scored', 'AI-written', 'Partly AI', 'Flagged (95% interval)', `p vs ${base.label} (adjusted)`].forEach(x => { const th = document.createElement('th'); th.textContent = x; hd.appendChild(th); });
  const b = t.createTBody();
  rows.forEach(r => { const tr = b.insertRow(); const c = v => { tr.insertCell().textContent = v; };
    c(r.year);
    if (r.status !== 'scored') { c('—'); c('—'); c('—'); c(r.note); c('—'); return; }
    c(r.n + (r.partial ? ' (partial)' : ''));
    c(`${r.ai} (${pct(r.ai, r.n)})`); c(`${r.mixed} (${pct(r.mixed, r.n)})`);
    c(`${pct(r.flag, r.n)} (${(100*r.lo).toFixed(1)}–${(100*r.hi).toFixed(1)})`);
    c(r.q === undefined ? 'baseline' : fmtP(r.q));
  });
  $('ch-sum').textContent = scored.map(r => `${r.year}: ${pct(r.flag, r.n)} flagged of ${r.n}`).join('; ') + '. 2022: no data. 2026: pending.';
})();

// chart
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function topRounded(x, y, w, h, r){ r = Math.min(r, h, w/2); return `M${x},${y+h} V${y+r} Q${x},${y} ${x+r},${y} H${x+w-r} Q${x+w},${y} ${x+w},${y+r} V${y+h} Z`; }
const tip = document.createElement('div'); tip.className = 'tip'; tip.setAttribute('role', 'tooltip');

function draw(){
  const box = $('chart'); box.replaceChildren(tip);
  const W = Math.max(300, box.clientWidth), H = W < 480 ? 270 : 300;
  const m = {l: 40, r: 8, t: 26, b: 46}, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const yMax = Math.max(0.2, Math.ceil((Math.max(...scored.map(r => r.hi)) + 0.05) * 10) / 10);
  const y = v => m.t + ph * (1 - v / yMax);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: W, height: H}, box);
  const cs = getComputedStyle(document.documentElement), C = n => cs.getPropertyValue(n).trim();
  for (let v = 0; v <= yMax + 1e-9; v += 0.1) {
    el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? C('--axis') : C('--grid'), 'stroke-width': 1, 'shape-rendering': 'crispEdges'}, svg);
    const t = el('text', {x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 12, fill: C('--mute')}, svg); t.textContent = Math.round(v * 100) + '%';
  }
  const band = pw / rows.length, bw = Math.min(24, band * 0.5);
  rows.forEach((r, i) => {
    const cx = m.l + band * (i + 0.5), x = cx - bw / 2;
    const yl = el('text', {x: cx, y: H - m.b + 18, 'text-anchor': 'middle', 'font-size': 13, fill: C('--ink'), 'font-weight': 600}, svg); yl.textContent = r.year;
    const nl = el('text', {x: cx, y: H - m.b + 34, 'text-anchor': 'middle', 'font-size': 11.5, fill: C('--mute')}, svg);
    nl.textContent = r.status === 'scored' ? `n=${r.n}` : r.status === 'nodata' ? 'no data' : 'pending';
    if (r.status !== 'scored') {
      const t = el('text', {x: cx, y: y(0) - 8, 'text-anchor': 'middle', 'font-size': 11.5, fill: C('--mute')}, svg); t.textContent = r.status === 'nodata' ? '—' : 'soon';
      return;
    }
    const fa = r.ai / r.n, fm = r.mixed / r.n, gap = 2;
    const hA = y(0) - y(fa), hM = y(0) - y(fm);
    if (fa > 0) el('path', {d: fm > 0 ? `M${x},${y(0)} H${x+bw} V${y(fa)} H${x} Z` : topRounded(x, y(fa), bw, hA, 4), fill: C('--s1')}, svg);
    if (fm > 0) { const top = y(fa + fm), h = hM - (fa > 0 ? gap : 0); if (h > 0) el('path', {d: topRounded(x, top, bw, h, 4), fill: C('--s2')}, svg); }
    // CI whisker on the total
    const wc = C('--ink2');
    el('line', {x1: cx, x2: cx, y1: y(r.lo), y2: y(r.hi), stroke: wc, 'stroke-width': 1.5}, svg);
    [r.lo, r.hi].forEach(v => el('line', {x1: cx - 4, x2: cx + 4, y1: y(v), y2: y(v), stroke: wc, 'stroke-width': 1.5}, svg));
    const lab = el('text', {x: cx, y: y(r.hi) - 7, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, fill: C('--ink')}, svg);
    lab.textContent = pct(r.flag, r.n, r.flag ? 1 : 0);
    // hit target: whole column band
    const hit = el('rect', {x: cx - band / 2, y: m.t, width: band, height: ph, class: 'hit go', tabindex: 0, role: 'button',
      'aria-label': `${r.year}: ${pct(r.flag, r.n)} flagged (${r.ai} AI-written, ${r.mixed} partly AI) of ${r.n}. Read this year's highlighted passages.`}, svg);
    hit.addEventListener('click', () => goYear(r.year));
    hit.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); goYear(r.year); } });
    const show = ev => {
      tip.replaceChildren();
      const h = document.createElement('div'); h.className = 'h'; h.textContent = `ASH ${r.year}` + (r.partial ? ' (partial sample)' : ''); tip.appendChild(h);
      [['AI-written', r.ai, '--s1'], ['Partly AI', r.mixed, '--s2']].forEach(([name, k, col]) => {
        const row = document.createElement('div'); row.className = 'row';
        const left = document.createElement('span'); const key = document.createElement('i'); key.className = 'key'; key.style.background = C(col);
        left.append(key, document.createTextNode(name));
        const v = document.createElement('b'); v.textContent = `${pct(k, r.n)} (${k})`; row.append(left, v); tip.appendChild(row);
      });
      const tot = document.createElement('div'); tot.className = 'row';
      const tl = document.createElement('span'); tl.textContent = 'Flagged, total';
      const tv = document.createElement('b'); tv.textContent = pct(r.flag, r.n); tot.append(tl, tv); tip.appendChild(tot);
      const mm = document.createElement('div'); mm.className = 'm';
      mm.textContent = `95% interval ${(100*r.lo).toFixed(1)}–${(100*r.hi).toFixed(1)}% · ${r.n} abstracts` + (r.q !== undefined ? ` · adjusted p vs ${base.label} ${fmtP(r.q)}` : ' · baseline year');
      tip.appendChild(mm);
      const go = document.createElement('div'); go.className = 'm'; go.textContent = 'Click to read its highlighted passages'; tip.appendChild(go);
      const bb = box.getBoundingClientRect();
      let px = (ev && ev.clientX ? ev.clientX - bb.left : cx) + 14, py = (ev && ev.clientY ? ev.clientY - bb.top : y(r.hi)) - 10;
      tip.style.opacity = 1;
      const tw = tip.offsetWidth; if (px + tw > bb.width) px = Math.max(0, (ev && ev.clientX ? ev.clientX - bb.left : cx) - tw - 14);
      tip.style.left = px + 'px'; tip.style.top = Math.max(0, py) + 'px';
    };
    const hide = () => { tip.style.opacity = 0; };
    hit.addEventListener('pointermove', show); hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', () => show(null)); hit.addEventListener('blur', hide);
  });
}
// ---- the passage explorer lives on its own page ----
const TERMS = DATA.terms;
function goYear(y){ location.href = 'passages/?year=' + y; }
(function(){
  const total = scored.reduce((s, r) => s + r.n, 0).toLocaleString('en-US');
  const nHl = scored.reduce((s, r) => s + DATA.hl[r.year], 0);
  const parts = scored.map(r => `${DATA.hl[r.year]} of ${r.n} from ${r.year}`).join(', ');
  $('ex-intro').textContent = `The detector highlighted text in ${nHl} of the ${total} abstracts scored (${parts}). The explorer shows each of those passages without titles or authors; the rest of each abstract is reduced to a word count. Filter by year or label, or search for a word.`;
})();
document.querySelectorAll('.era').forEach(e => { e.textContent = `${TERMS.rule.era_from}–${TERMS.latest}`; });

// ---- words and phrases ----
(function(){
  const R = TERMS.rule, yrs = TERMS.years, SHOW = 12;
  let kind = 'words', all = false;
  const nf = x => x.toLocaleString('en-US');
  $('w-rule').textContent = `Listed when a term passes two tests after correction for the number of terms tested (Benjamini–Hochberg, q < ${R.q}): it is more common in the highlighted passages than in the unhighlighted text of the same ${R.era_from}–${TERMS.latest} abstracts (${nf(R.n_hl_tests)} terms tested), and more ${TERMS.latest} abstracts use it than ${TERMS.base_label} abstracts (${nf(R.n_year_tests)} terms tested). It must also turn up in at least ${R.abstracts} different highlighted abstracts and be more than ${R.ratio} times as common in the highlighted passages as in the unhighlighted text. Under each ratio: the first test's q-value, and how many times the term was used in highlighted and in other text (highlighted text is ${R.hl_share}% of the words in those abstracts). Each line is the share of sampled abstracts using the term, by year, on its own scale; the dashed stretch spans 2022, which has no data. The list is exploratory (see "Words and phrases" under What was done). Select a term to read the passages that contain it.`;
  const fq = q => { if (q < 0.001) return 'q < 0.001';
    let s = q.toPrecision(2); if (+s >= R.q && q < R.q) s = q.toPrecision(3); return 'q = ' + s; };
  function spark(v){
    const W = 84, H = 24, x0 = yrs[0], x1 = yrs[yrs.length - 1], mx = Math.max(...v.s, 0.1);
    const X = y => 3 + (W - 6) * (y - x0) / (x1 - x0), Y = p => H - 2 - (H - 6) * p / mx;
    const svg = el('svg', {width: W, height: H, viewBox: `0 0 ${W} ${H}`, 'aria-hidden': 'true'});
    el('line', {x1: 0, x2: W, y1: H - 1.5, y2: H - 1.5, style: 'stroke:var(--axis)', 'stroke-width': 1}, svg);
    for (let i = 1; i < yrs.length; i++) {
      const a = {x1: X(yrs[i-1]), y1: Y(v.s[i-1]), x2: X(yrs[i]), y2: Y(v.s[i]), style: 'stroke:var(--ink2)', 'stroke-width': 2, 'stroke-linecap': 'round'};
      if (yrs[i] - yrs[i-1] > 1) { a['stroke-dasharray'] = '1.5 3.5'; a['stroke-width'] = 1.5; }
      el('line', a, svg);
    }
    el('circle', {cx: X(x1), cy: Y(v.s[v.s.length - 1]), r: 3, style: 'fill:var(--ink);stroke:var(--panel)', 'stroke-width': 1.5}, svg);
    return svg;
  }
  function render(){
    const list = TERMS[kind], t = $('wt'); t.replaceChildren();
    const hd = t.createTHead().insertRow();
    [kind === 'words' ? 'Word' : 'Phrase', 'Times as common highlighted', `Abstracts using it, ${TERMS.base_label.replace(/–20(\d\d)$/, '–$1')} → ${TERMS.latest}`].forEach((x, i) => {
      const th = document.createElement('th'); th.textContent = x; if (i) th.style.textAlign = 'right'; hd.appendChild(th); });
    const b = t.createTBody();
    list.slice(0, all ? list.length : SHOW).forEach(v => {
      const tr = b.insertRow();
      const btn = document.createElement('a'); btn.className = 'termbtn'; btn.textContent = v.t;
      btn.href = 'passages/?q=' + encodeURIComponent(v.t) + '&exact=1';
      btn.setAttribute('aria-label', `${v.t}: read the passages that contain it`);
      tr.insertCell().appendChild(btn);
      const c2 = tr.insertCell(); const r = document.createElement('b'); r.textContent = `${v.r.toFixed(1)}×`;
      const k = document.createElement('span'); k.className = 'sub2'; k.textContent = `${fq(v.qb)} · ${v.h} vs ${v.u} uses`;
      const ka = document.createElement('span'); ka.className = 'nabs'; ka.textContent = ` · ${v.k} abstracts`; k.appendChild(ka); c2.append(r, k);
      const c3 = tr.insertCell(); const d = document.createElement('div'); d.className = 'spark';
      const nums = document.createElement('span'); nums.textContent = `${v.b.toFixed(1)}% → ${v.s[v.s.length - 1].toFixed(1)}%`;
      const qy = document.createElement('span'); qy.className = 'sub2'; qy.textContent = fq(v.qa); nums.appendChild(qy);
      d.append(spark(v), nums); c3.appendChild(d);
      d.title = 'Share of sampled abstracts using it\n' + yrs.map((y, i) => `${y}: ${v.s[i]}%`).join('\n');
      c3.setAttribute('aria-label', 'Share of abstracts using it: ' + yrs.map((y, i) => `${y} ${v.s[i]}%`).join(', '));
    });
    const more = $('wt-more');
    more.hidden = list.length <= SHOW;
    more.textContent = all ? 'Show fewer' : `Show all ${list.length} ${kind}`;
  }
  const tabs = {words: $('tab-words'), phrases: $('tab-phrases')};
  Object.entries(tabs).forEach(([k, btn]) => btn.addEventListener('click', () => {
    kind = k; all = false; Object.entries(tabs).forEach(([k2, b2]) => b2.setAttribute('aria-selected', String(k2 === k))); render(); }));
  $('wt-more').onclick = () => { all = !all; render(); };
  render();
})();

draw();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""

PAS_BODY = r"""<body>
<div class="wrap">
<a class="back" href="../">&larr; How much ASH abstract text is AI-written?</a>
<h1>Highlighted passages from ASH abstracts</h1>
<p class="sub">Every passage an AI-text detector highlighted in random samples of American Society of Hematology annual-meeting abstracts.</p>
<div class="chips"><span class="chip warn">Preliminary pilot &middot; not peer reviewed</span><span class="chip" id="upd"></span><span class="chip" id="mdl"></span></div>

<section class="card" id="ex" aria-labelledby="ex-h">
  <h2 id="ex-h">Read the highlighted passages</h2>
  <p class="cap" id="ex-intro"></p>
  <div class="controls">
    <select id="f-y" aria-label="Meeting year"></select>
    <select id="f-l" aria-label="Detector's label for the whole abstract">
      <option value="">All abstract labels</option>
      <option value="AI">AI-written</option>
      <option value="Mixed">Partly AI-written</option>
      <option value="Human">Human-written overall</option>
    </select>
    <select id="f-s" aria-label="Order">
      <option value="most">Most highlighted first</option>
      <option value="least">Least highlighted first</option>
      <option value="shuffle">Shuffled</option>
    </select>
    <input type="search" id="f-q" placeholder="Find a word or phrase" aria-label="Find a word or phrase in the passages" autocomplete="off">
  </div>
  <div class="legend"><span><mark class="g">AI-generated</mark></span><span><mark class="a">AI-assisted</mark></span><span><span class="find">search match</span></span></div>
  <p class="count" id="ex-cnt" aria-live="polite"></p>
  <div id="ex-list"></div>
  <div class="pager"><button id="ex-prev">Previous</button><span id="ex-pg"></span><button id="ex-next">Next</button></div>
</section>

<section class="card prose" aria-labelledby="a-h">
  <h2 id="a-h">How to read these</h2>
  <ul>
    <li>A highlight is the detector's judgment, not proof that AI wrote the text, and it says nothing about whether the science is sound. Using AI to polish English is common and usually allowed.</li>
    <li>The detector judges each stretch of an abstract on its own. Solid highlights are passages it labeled AI-generated; lighter, dotted ones are AI-assisted. The label next to each abstract is its verdict on the abstract as a whole.</li>
    <li>Passages are shown without titles, authors, abstract numbers or links, and not in abstract-number order. Trial registration numbers and pasted title lines are masked. Passages can still name an institution, a study or a data set.</li>
    <li>Methods, statistics and the words these passages lean on are on the <a href="../">main page</a>.</li>
  </ul>
</section>

<p class="fine">Independent analysis by Eric Perkey, Frank Wen, Reid Shaw and Max Wen; views are the authors' own. Not affiliated with or endorsed by the American Society of Hematology or Pangram Labs. Abstract text &copy; the American Society of Hematology; passages the detector highlighted are quoted for research and commentary.</p>
<p class="fine">AI use: the language on this page was generated by Claude (Anthropic) at the authors' direction, and Claude helped write the analysis code. Pangram scored the abstracts. The authors are responsible for the content.</p>
</div>

<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
$('upd').textContent = 'Updated ' + DATA.updated;
$('mdl').textContent = DATA.model;
const scored = DATA.rows, PAS = DATA.passages;
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function makeRe(q, exact, flags){
  q = q.trim(); if (!q) return null;
  // word boundaries only next to word characters, so searches like "≥3" still match
  const pre = /^\w/.test(q) ? '\\b' : '', post = exact && /\w$/.test(q) ? '\\b' : '';
  return new RegExp(pre + q.split(/\s+/).map(esc).join('\\s+') + post, flags);
}
function addText(parent, s, re){
  if (!re) { parent.appendChild(document.createTextNode(s)); return; }
  re.lastIndex = 0; let last = 0, m;
  while ((m = re.exec(s))) {
    if (!m[0]) { re.lastIndex++; continue; }
    if (m.index > last) parent.appendChild(document.createTextNode(s.slice(last, m.index)));
    const f = document.createElement('span'); f.className = 'find'; f.textContent = m[0]; parent.appendChild(f);
    last = m.index + m[0].length;
  }
  if (last < s.length) parent.appendChild(document.createTextNode(s.slice(last)));
}
const smooth = () => matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
const toExplorer = () => $('ex').scrollIntoView({behavior: smooth(), block: 'start'});

const EX = {page: 0, rows: [], exact: false, re: null, PER: 10};
const LAB = {AI: ['AI-written', '--s1'], Mixed: ['partly AI-written', '--s2'], Human: ['human-written overall', '--axis']};
PAS.forEach((a, i) => { a.i = i; a.cov = a.nw ? a.hw / a.nw : 0;
  a.txt = a.p.filter(Array.isArray).map(b => b.map(s => s[1]).join('')).join('\n'); });
const scoredN = Object.fromEntries(scored.map(r => [r.year, r.n]));
const withHl = y => PAS.filter(a => a.y === y).length;
(function(){
  const total = scored.reduce((s, r) => s + r.n, 0).toLocaleString('en-US');
  const parts = scored.map(r => `${withHl(r.year)} of ${r.n} from ${r.year}`).join(', ');
  $('ex-intro').textContent = `The detector highlighted text in ${PAS.length} of the ${total} abstracts scored (${parts}). Only the highlighted passages are shown; the rest of each abstract is reduced to a word count.`;
  const fy = $('f-y'); const o = document.createElement('option'); o.value = ''; o.textContent = 'All years'; fy.appendChild(o);
  scored.forEach(r => { const op = document.createElement('option'); op.value = r.year; op.textContent = `${r.year} (${withHl(r.year)} with highlights)`; fy.appendChild(op); });
})();
// filters can arrive in the link (from the main page) and are kept in the address bar
const P = new URLSearchParams(location.search);
if ([...$('f-y').options].some(o => o.value && o.value === P.get('year'))) $('f-y').value = P.get('year');
if (['AI', 'Mixed', 'Human'].includes(P.get('label'))) $('f-l').value = P.get('label');
if (['most', 'least', 'shuffle'].includes(P.get('order'))) $('f-s').value = P.get('order');
if (P.get('q')) { $('f-q').value = P.get('q').slice(0, 100); EX.exact = P.get('exact') === '1'; }
function syncUrl(){
  const u = new URLSearchParams();
  if ($('f-y').value) u.set('year', $('f-y').value);
  if ($('f-l').value) u.set('label', $('f-l').value);
  if ($('f-s').value !== 'most') u.set('order', $('f-s').value);
  const q = $('f-q').value.trim(); if (q) { u.set('q', q); if (EX.exact) u.set('exact', '1'); }
  const s = u.toString();
  try { history.replaceState(null, '', s ? '?' + s : location.pathname); } catch (e) {}
}
function filtered(){ return !!($('f-y').value || $('f-l').value || $('f-q').value.trim()); }
function applyEx(){
  const y = $('f-y').value, l = $('f-l').value, s = $('f-s').value, q = $('f-q').value;
  const re = makeRe(q, EX.exact, 'i'); EX.re = makeRe(q, EX.exact, 'gi');
  const rows = PAS.filter(a => (!y || String(a.y) === y) && (!l || a.lab === l) && (!re || re.test(a.txt)));
  if (s === 'most') rows.sort((a, b) => b.cov - a.cov || a.i - b.i);
  else if (s === 'least') rows.sort((a, b) => a.cov - b.cov || a.i - b.i);
  else rows.sort((a, b) => a.i - b.i);
  EX.rows = rows; EX.page = 0; renderEx(); syncUrl();
}
function card(a){
  const art = document.createElement('article'); art.className = 'ab';
  const h = document.createElement('div'); h.className = 'ab-h';
  const yb = document.createElement('b'); yb.textContent = `ASH ${a.y}`;
  const lab = document.createElement('span'); lab.className = 'lab';
  const key = document.createElement('i'); key.style.background = `var(${LAB[a.lab][1]})`;
  lab.append(key, document.createTextNode(`Labeled ${LAB[a.lab][0]}`));
  const cv = document.createElement('span'); const pc = 100 * a.cov;
  cv.textContent = `${pc > 0 && pc < 1 ? '<1' : Math.round(pc)}% of ${a.nw} words highlighted`;
  h.append(yb, lab, cv); art.appendChild(h);
  a.p.forEach(it => {
    if (typeof it === 'number') {
      const g = document.createElement('div'); g.className = 'gap'; g.textContent = `${it} word${it === 1 ? '' : 's'} not highlighted`; art.appendChild(g); return; }
    const p = document.createElement('p'); p.className = 'ptext';
    it.forEach(([code, text]) => { const m = document.createElement('mark'); m.className = code; addText(m, text, EX.re); p.appendChild(m); });
    const cap = document.createElement('p'); cap.className = 'pc';
    cap.textContent = [...new Set(it.map(([code, , conf]) => (code === 'g' ? 'AI-generated' : 'AI-assisted') + (conf.length ? ` · ${conf.join(' and ')} confidence` : '')))].join('; ');
    art.append(p, cap);
  });
  return art;
}
function renderEx(){
  const L = $('ex-list'), n = EX.rows.length, pages = Math.max(1, Math.ceil(n / EX.PER));
  EX.page = Math.max(0, Math.min(EX.page, pages - 1));
  const from = EX.page * EX.PER;
  L.replaceChildren(...EX.rows.slice(from, from + EX.PER).map(card));
  if (!n) {
    const e = document.createElement('p'); e.className = 'empty'; const y = $('f-y').value;
    e.textContent = y && !withHl(+y) && !$('f-l').value && !$('f-q').value.trim()
      ? `The detector highlighted nothing in the ${scoredN[y]} ASH ${y} abstracts scored.`
      : 'No highlighted passage matches these filters.';
    L.appendChild(e);
  }
  const c = $('ex-cnt'); c.replaceChildren(n ? `Showing ${from + 1}–${Math.min(n, from + EX.PER)} of ${n} abstract${n === 1 ? '' : 's'}` : '');
  if (filtered()) { const clr = document.createElement('button'); clr.className = 'linkbtn'; clr.textContent = 'Clear filters'; clr.onclick = clearEx; c.append(clr); }
  $('ex-pg').textContent = `Page ${EX.page + 1} of ${pages}`;
  $('ex-prev').disabled = EX.page === 0; $('ex-next').disabled = EX.page >= pages - 1;
  document.querySelector('.pager').hidden = pages === 1;
}
function clearEx(){ $('f-y').value = ''; $('f-l').value = ''; $('f-q').value = ''; EX.exact = false; applyEx(); }
['f-y', 'f-l', 'f-s'].forEach(id => $(id).addEventListener('change', applyEx));
let tq; $('f-q').addEventListener('input', () => { EX.exact = false; clearTimeout(tq); tq = setTimeout(applyEx, 180); });
$('ex-prev').onclick = () => { EX.page--; renderEx(); toExplorer(); };
$('ex-next').onclick = () => { EX.page++; renderEx(); toExplorer(); };
applyEx();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
