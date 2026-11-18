# Artificial ASH: a free, re-runnable audit of AI-written text in ASH abstracts

This repository re-runs the analysis behind
[eperkey.org/artificial_ash](https://eperkey.org/artificial_ash/): a commercial AI-text detector
(Pangram 4.1) applied to random samples of 500 American Society of Hematology (ASH) annual-meeting
abstracts from each of 2020, 2021, 2023, 2024 and 2025, compared with the pre-ChatGPT years 2020–2021.

It is a "compute capsule" that costs nothing to run: the code, a pinned software environment, and
checks that tell you whether you got the same answer. It starts from the abstracts as ASH publishes
them and ends with the study's statistics, tables and figures. It runs on your own computer with
Docker or Python, in a browser on [mybinder.org](https://mybinder.org/), or on GitHub's free runners.
A copy of each release, the replay files and the outputs are also at
[eperkey.org/artificial_ash/capsule/](https://eperkey.org/artificial_ash/capsule/).

## What you can reproduce, and what you need

| Step | What it does | Needs | Expected result |
|---|---|---|---|
| `data` | Downloads every ASH abstract for each year from Crossref, cleans it, draws the random samples (seed 20261113) | Internet, a few minutes, ~400 MB of disk | 5 sample files in `data/samples/` |
| `verify` | Checks your abstracts against the study's SHA-256 fingerprints, abstract by abstract | `data` | `VERIFIED: same abstracts as the study` |
| `replay` | Downloads the study's Pangram results with the abstract text removed (3.5 MB) and checks each file's SHA-256 | Internet | `REPLAY FILES OK` |
| `rehydrate` | Puts your downloaded text back into the replay files, giving the Pangram results exactly as returned | `data`, `replay` | `REHYDRATED: 2500 responses, every text hash matched` |
| `score` | Scores the samples again with Pangram yourself, instead of `replay` and `rehydrate` (`--max-usd` caps the total; it refuses to run over rehydrated results) | `data`, a paid Pangram API key (2,500 abstracts, 1.43 million words; about $775 at list price, because Pangram bills each started 100 words of every abstract) | new Pangram results |
| `analyze` | Statistics, tables and figures (below), each compared with the published build | results from `rehydrate` or `score` | `RESULTS MATCH the published counts`, then `IDENTICAL` for every output and `ANALYSIS REPRODUCED` |
| `selftest` | Offline check, no data needed: recomputes the published q-values, tests the replay round trip, and redraws the three figures from the published counts | nothing | `SELFTEST PASSED`, `TEST PASSED`, `FIGURE SELFTEST PASSED` |

`all` runs `data`, `verify`, `replay`, `rehydrate` and `analyze` in a row.

## What `analyze` produces

| File | What it is |
|---|---|
| `results/figures/figure1_flagged_by_year.svg` (+ `.pdf`, `.png`) | Share of abstracts the detector labeled AI-written or partly AI-written, by meeting year, with 95% intervals and q-values |
| `results/figures/figure2_ai_rewrites.svg` (+ `.pdf`, `.png`) | How the detector labeled the 2020 abstracts as written and after an AI model rewrote them |
| `results/figures/figure3_ai_written_controls.svg` (+ `.pdf`, `.png`) | How the detector labeled 100 abstracts an AI model wrote from notes on 2025 abstracts, overall and by the label of the abstract each was based on |
| `results/tables/by_year.csv` | Figure 1's numbers: labels per year, share flagged, Wilson 95% interval, Fisher exact test against pooled 2020–2021, Benjamini–Hochberg q |
| `results/tables/terms.csv` | Words and phrases that pass both term tests: more common in the text the detector highlighted, and more common in 2025 than in 2020–2021 |
| `results/tables/term_tests.json` | The rule for listing a term and how many terms each test covered |
| `results/tables/ai_rewrites.csv` | Figure 2's numbers |
| `results/tables/ai_written_controls.csv` | Figure 3's numbers |
| `results/pilot_summary.md` | Per-year summary, including the other outcomes and the 2023–2025 trend |

The tables, the SVG figures and the summary are byte-identical on every platform, and `analyze`
checks them against `manifests/expected_outputs.json`. The PDF and PNG copies are for convenience
and are not compared. The capsule does not build the public web page: the page's text is not part of
the analysis.

## Run it

**Docker** (the exact environment of the study). Create the folders first and run the container as
yourself; otherwise, on Linux, Docker creates them owned by root and the steps cannot write to them:

```sh
docker build -t artificial-ash .
mkdir -p data replay results
run() { docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/capsule/data" \
  -v "$PWD/replay:/capsule/replay" -v "$PWD/results:/capsule/results" artificial-ash "$@"; }
run selftest
run all          # data, verify, replay, rehydrate, analyze; figures land in ./results/figures
```

On Windows with Docker Desktop (PowerShell), leave out `--user ...` and write `${PWD}` for `$PWD`.

**Binder** (free, in your browser): [launch it on mybinder.org](https://mybinder.org/v2/gh/etperkey/artificial-ash/HEAD),
open a terminal (File > New > Terminal), and run `python -I reproduce.py selftest`, then
`python -I reproduce.py all`. The figures appear in `results/figures/`.

**Plain Python 3.12**:

```sh
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --require-hashes --no-deps -r requirements.txt
python -I reproduce.py selftest
python -I reproduce.py all
```

## What is not in this repository, and why

* **The abstracts.** ASH owns the copyright in its meeting abstracts, so they are not redistributed.
  `data` downloads them from Crossref, where ASH publishes them, and `verify` proves they are the
  same text the study analyzed (`manifests/sample_<year>.tsv` lists each sampled DOI with the
  SHA-256 of its cleaned text). If Crossref corrects a sampled abstract, `verify` names it. If
  Crossref adds or removes an abstract, the random draw shifts and `verify` reports the whole
  year's sample as different.
* **Pangram's responses as returned.** Each response repeats the abstract it scored. The replay
  files keep everything else (labels, scores and the start and end of every judged passage) plus
  hashes, and `rehydrate` puts your own downloaded text back, refusing if a single character differs.
  The replay files hold no abstract text. They are published at
  [eperkey.org/artificial_ash/capsule/](https://eperkey.org/artificial_ash/capsule/)
  and `replay` downloads them; `manifests/replay_sha256.txt` pins their contents. They list each
  sampled abstract's DOI with the detector's label. A detector label is a statistical estimate, not
  proof that any author used AI, and the study reports only aggregates.
* **A Pangram API key.** Pangram is a commercial detector; re-scoring costs money. The study used
  the `pangram-4.1` model throughout, and Pangram may retire it, which is why the original responses
  are kept.

## Layout

```
reproduce.py                 one entry point for every step
scripts/fetch_crossref.py    download one year's ASH abstracts (Blood supplement) from Crossref
scripts/clean.py             keep only the abstract body, by the same rules every year
scripts/sample.py            seeded simple random sample
scripts/fingerprint.py       write or check the abstract fingerprints
scripts/score_pangram.py     Pangram API client (resumable, budget-capped)
scripts/pangram_replay.py    fetch the replay files; strip text from Pangram responses / restore it
scripts/summarize.py         per-year summary (results/pilot_summary.md)
scripts/analysis.py          statistics, tables and figures (results/tables, results/figures)
scripts/check_results.py     compare a run with manifests/results_summary.csv (independent re-implementation)
scripts/validation_summary.py  Figure 2's counts from the AI-rewrite run (needs that run's files)
scripts/validation_fullgen_summary.py  Figure 3's counts from the AI-written control run (needs that run's files)
manifests/                   fingerprints, replay checksums, published counts, expected output hashes
Dockerfile, requirements.txt, binder/   the pinned environment (Python 3.12.10, hash-locked packages)
```

## Environment

Python 3.12.10 in a Docker base image pinned by digest; NumPy 2.3.5, SciPy 1.17.0, statsmodels
0.14.6, matplotlib 3.10.8 and their dependencies pinned by version and hash. Figures use the DejaVu
Sans font that ships with matplotlib, so they look the same on every system. GitHub Actions rebuilds
the image and runs the self-test on every change (which includes redrawing the figures on Linux and
comparing them byte for byte), and re-downloads and verifies the abstracts once a month.

## The second figure (AI rewrites)

The second figure tests the detector on AI-rewritten abstracts: each of the first 250 abstracts of the 2020
sample was sent once to Claude Opus 5.5 (Anthropic Message Batches API, default settings, no system
prompt) with the instruction quoted in `manifests/validation_rewrite.json`; 12 requests were declined
by Anthropic's safety filter and the 238 rewrites were scored with pangram-4.1. The rewrites are copies
of ASH's abstracts, so they are not published; the manifest holds the counts the figure shows, and
`scripts/validation_summary.py` is the script that produced it from the run's files. Re-running it
needs your own Anthropic and Pangram keys (about $5 and $65 at list prices).

## The third figure (AI-written controls)

The third figure tests the detector on abstracts written entirely by AI. The first 115 abstracts of
the 2025 sample (it is in random order) were each sent to Claude Opus 5.5 twice (Anthropic Messages
API, default settings, no system prompt): once to turn the abstract into notes, and once, in a
separate request that never saw the abstract, to write a new abstract from the notes alone. Both
instructions are quoted in `manifests/validation_fullgen.json`. Anthropic's safety filter declined 10
abstracts at one of the two steps; the controls are the first 100 of the other 105, a rule fixed
before any was scored. They were scored with pangram-4.1: 20 labeled AI-written, 37 partly
AI-written, 43 human-written. The label followed the label of the abstract a control was based on
(none of the 69 controls based on abstracts labeled human-written was labeled AI-written; 7 of the 8
based on abstracts labeled AI-written were), which the study has not explained. The controls are
derived from ASH's abstracts, so they are not published; the manifest holds the counts the figure
shows, and `scripts/validation_fullgen_summary.py` produced it from the run's files. Re-running it
needs your own Anthropic and Pangram keys (about $10 and $32 at list prices).

## Changes

* **2026-11-18, v0.3.0: 500 abstracts a year; AI-written controls.** The other 250 abstracts of each
  year's seeded sample of 500 were scored (same detector, same settings), so every sampled abstract
  is now scored: 2,500 in all. Flagged: 0 of 1,000 in 2020-2021, 8 of 500 (1.6%) in 2023, 38 (7.6%)
  in 2024 and 168 (33.6%) in 2025. The samples and their fingerprints are unchanged. A third figure
  and table report the detector on 100 abstracts written entirely by AI (above). Tables now write
  very small p- and q-values in scientific notation; v0.2.0 wrote values below 1e-12 as 0. The
  replay files at eperkey.org now cover all 2,500 abstracts, so v0.2.0's `replay` step no longer
  finds its files there: they are kept at `capsule/replay/v0.2.0/`, and v0.2.0 can fetch them with
  `python -I scripts/pangram_replay.py fetch --url https://eperkey.org/artificial_ash/capsule/replay/v0.2.0/`.

* **2026-11-15, v0.2.0: analysis outputs; public replay files.** `analyze` now writes the figures and
  tables (`scripts/analysis.py`) instead of rebuilding the web page, and the page builder is no
  longer part of the capsule. The statistics are unchanged. The replay files are published, so the
  whole analysis can be re-run without asking the authors; `replay` downloads them and `all`
  includes it. matplotlib is added to the pinned environment. The ASH_BUILD_DATE setting is gone:
  no output carries a date.
* **2026-11-14, cleaning fix.** The 2025 export often runs section labels into the first sentence
  ("Background Acute…"); the first cleaning rules missed these in about 19% of 2025 abstracts
  against 1–2% in other years. `clean.py` now removes them, plus pasted title lines and short label
  lines. The samples are unchanged (same abstracts, same order); the 87 sampled abstracts whose text
  changed were scored again. 86 kept their label; one 2025 abstract moved from AI to Mixed, so 2025
  is 29 AI and 47 Mixed (76 flagged either way). The fingerprints, replay files and expected outputs
  here are for the corrected text; the original results are archived by the authors.

## Citation and license

See `CITATION.cff`. Code: MIT license. Abstracts: copyright the American Society of Hematology.
This is an independent analysis, not affiliated with or endorsed by ASH or Pangram Labs.

AI use: the language in this README and on the public page was generated by Claude (Anthropic) at the
authors' direction, and Claude helped write the code. Pangram scored the abstracts. The authors are
responsible for the content.
