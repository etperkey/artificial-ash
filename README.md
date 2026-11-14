# Artificial ASH: a free, re-runnable audit of AI-written text in ASH abstracts

This repository re-runs the analysis behind
[eperkey.org/artificial_ash](https://eperkey.org/artificial_ash/): a commercial AI-text detector
(Pangram 4.1) applied to random samples of 250 American Society of Hematology (ASH) annual-meeting
abstracts from each of 2020, 2021, 2023, 2024 and 2025, compared with the pre-ChatGPT years 2020–2021.

It is a "compute capsule" that costs nothing to run: the code, a pinned software environment, and
checks that tell you whether you got the same answer. It runs on your own computer with Docker or
Python, in a browser on [mybinder.org](https://mybinder.org/), or on GitHub's free runners.

## What you can reproduce, and what you need

| Step | What it does | Needs | Expected result |
|---|---|---|---|
| `data` | Downloads every ASH abstract for each year from Crossref, cleans it, draws the random samples (seed 20261113) | Internet, about 30 minutes, ~270 MB | 5 sample files in `data/samples/` |
| `verify` | Checks your abstracts against the study's SHA-256 fingerprints, abstract by abstract | `data` | `VERIFIED: same abstracts as the study` |
| `rehydrate` | Rebuilds the Pangram results from the replay files (text stripped, see below) | `data`, the replay files | `REHYDRATED: 1250 responses, every text hash matched` |
| `score` | Scores the samples again with Pangram yourself | `data`, a paid Pangram API key (~715,000 words, about $360 at list price) | new Pangram results |
| `analyze` | Statistics, the two public pages (results, and the passage explorer), and a comparison with the published numbers | results from `rehydrate` or `score` | `RESULTS MATCH the published counts`; with `ASH_BUILD_DATE=2026-11-14`, both pages are byte-identical to the published build |
| `selftest` | Offline check of the environment: recomputes the published q-values and tests the replay round trip | nothing | `SELFTEST PASSED`, `TEST PASSED` |

`all` runs `data`, `verify`, `rehydrate` and `analyze` in a row.

## Run it

**Docker** (the exact environment of the study):

```sh
docker build -t artificial-ash .
docker run --rm artificial-ash selftest
docker run --rm -v "$PWD/data:/capsule/data" artificial-ash data
docker run --rm -v "$PWD/data:/capsule/data" artificial-ash verify
# with the replay files in ./replay:
docker run --rm -e ASH_BUILD_DATE=2026-11-14 -v "$PWD/data:/capsule/data" \
  -v "$PWD/replay:/capsule/replay:ro" -v "$PWD/results:/capsule/results" \
  -v "$PWD/public_dashboard:/capsule/public_dashboard" artificial-ash all
```

**Binder** (free, in your browser): [launch it on mybinder.org](https://mybinder.org/v2/gh/etperkey/artificial-ash/HEAD),
open a terminal (File > New > Terminal), and run `python -I reproduce.py selftest`, then `data` and `verify`.

**Plain Python 3.12**:

```sh
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --require-hashes --no-deps -r requirements.txt
python -I reproduce.py selftest
python -I reproduce.py data
python -I reproduce.py verify
```

## What is not in this repository, and why

* **The abstracts.** ASH owns the copyright in its meeting abstracts, so they are not redistributed.
  `data` downloads them from Crossref, where ASH publishes them, and `verify` proves they are the
  same text the study analyzed (`manifests/sample_<year>.tsv` lists each sampled DOI with the
  SHA-256 of its cleaned text). If Crossref ever corrects an abstract, `verify` names it.
* **Pangram's responses as returned.** Each response repeats the abstract it scored. The replay
  files keep everything else (labels, scores and the start and end of every judged passage) plus
  hashes, and `rehydrate` puts your own downloaded text back, refusing if a single character differs.
  The replay files hold no abstract text. They are available from the authors on request.
* **A Pangram API key.** Pangram is a commercial detector; re-scoring costs money. The study used
  the `pangram-4.1` model throughout, and Pangram may retire it, which is why the original responses
  are kept.

## Layout

```
reproduce.py              one entry point for every step
scripts/fetch_crossref.py download one year's ASH abstracts (Blood supplement) from Crossref
scripts/clean.py          keep only the abstract body, by the same rules every year
scripts/sample.py         seeded simple random sample
scripts/score_pangram.py  Pangram API client (resumable, budget-capped)
scripts/pangram_replay.py strip text from Pangram responses / restore it from your download
scripts/fingerprint.py    write or check the abstract fingerprints
scripts/summarize.py      per-year summary (results/pilot_summary.md)
scripts/build_dashboard.py the public pages (public_dashboard/index.html and passages/index.html)
scripts/check_results.py  compare a run with manifests/results_summary.csv
manifests/                fingerprints, published counts and q-values, expected output hashes
Dockerfile, requirements.txt, binder/   the pinned environment (Python 3.12.10, hash-locked packages)
```

## Environment

Python 3.12.10 in a Docker base image pinned by digest; NumPy 2.3.5, SciPy 1.17.0, statsmodels
0.14.6 and their dependencies pinned by version and hash. GitHub Actions rebuilds the image and runs
the self-test on every change, and re-downloads and verifies the abstracts once a month.

## Citation and license

See `CITATION.cff`. Code: MIT license. Abstracts: copyright the American Society of Hematology.
This is an independent analysis, not affiliated with or endorsed by ASH or Pangram Labs.

AI use: the language in this README and on the public page was generated by Claude (Anthropic) at the
authors' direction, and Claude helped write the code. Pangram scored the abstracts. The authors are
responsible for the content.
