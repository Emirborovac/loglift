# LogLift

**An open benchmark and baseline for digitizing scanned paper well logs — and an open challenge to solve the hard part: curve extraction.**

Millions of oil & gas wells were logged on paper before the digital era. Those measurements survive only as scanned images that no software can read. Converting them is a real, unsolved problem — and there has never been a large, open dataset to work on it. **LogLift is that dataset, plus a working baseline, plus a scoreboard.**

---

## 🎯 The open challenge: curve extraction

Reading the **depth scale** off a scanned log is essentially solved here (see results below). **Tracing the curves is not — and nobody has solved it, anywhere.** The best published method reaches only ~0.62 correlation on gamma ray, the *easiest* curve. Sharp peaks look identical to grid lines; washed-out zones are a scribble even a human struggles with.

**That's the invitation.** We provide the data, a baseline, and a reproducible benchmark. Beat it. Every improvement to curve extraction is a real contribution to a problem the whole industry keeps a human in the loop for.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to run the benchmark and submit results, and **[LEADERBOARD.md](LEADERBOARD.md)** for current scores.

---

## What's in here

| Component | What it is |
|---|---|
| **Open benchmark** | 22,490 wells with a paired scanned image **and** verified digital LAS — free, no login. Built from public Kansas Geological Survey data. |
| **Trained depth reader** | A small CRNN that reads depth-scale numbers off the scan, trained with *zero manual labels* (self-supervised), ~90% exact-match on held-out wells. |
| **Baseline pipeline** | layout detection → depth calibration → curve tracing → LAS export, end to end. |
| **Assisted app** | Upload a scan, auto-trace, drag curves to correct, export LAS (human-in-the-loop, the way real tools work). |
| **Reproducible benchmark** | Score any pipeline against ground truth on 13,000+ wells with one command. |

## Current results (baseline, 13,112 wells)

| Metric | Result | Notes |
|---|---|---|
| **Depth auto-conversion** | **77.7% of wells** | the well is fully depth-registered, hands-off |
| **Depth accuracy** | **0.82 ft median RMS** | on par with commercial vendors; refuses rather than fabricates |
| Depth-reader model | ~90% exact-match | held-out wells, self-supervised |
| **Curve extraction** | 0.51 median best-curve correlation | **← the open challenge.** SOTA is ~0.62 |

Depth is production-grade. **Curves are the frontier — that's where contributors come in.**

## Quick start

```bash
git clone https://github.com/Emirborovac/loglift.git && cd loglift
pip install -r requirements.txt

# build the paired dataset from public KGS data (downloads free, no login)
python -m pipeline.indexes      # join scan & LAS indexes -> 22,490 pairs
python -m pipeline.pairs        # match each scan to its LAS depth interval
python -m pipeline.download --limit 200   # grab 200 pairs to start (~1 GB)

# run the baseline end-to-end on one scan
python convert.py data/pairs/<well>/scan_*.tif

# score the baseline against ground truth
python benchmark.py --workers 4

# try the assisted app
python -m uvicorn app.main:app --port 8517   # then open localhost:8517
```

## How it works

1. **Layout detection** — find the log section, curve tracks, and depth column
2. **Depth calibration** — OCR + a trained CRNN read the depth labels; a RANSAC fit maps image rows to feet (physics-gated, refuses when unsure)
3. **Curve tracing** — the baseline traces each curve (the part we want you to improve)
4. **Header reading** — curve names + engineering-unit scales
5. **LAS export** — standard LAS 2.0 any petrophysics software reads

A `playground/` app lets you iterate on the curve algorithm visually on a single log.

## Why depth works but curves don't (yet)

The depth reader trains itself: the pipeline can *verify* a depth label (round number, on a straight line, physical scale), so it self-labels 120,000 examples with no human. Curves have no such clean self-check — a sharp peak and a grid line are locally identical. That asymmetry is the whole story, and the open problem.

## Roadmap / where to help

- **Curve segmentation model** (UNet on synthetic + self-labeled tracks — scaffolding is in `training/`)
- Sharp-peak vs grid-line disambiguation
- Overlapping / crossing curve separation
- International generalization (Dutch NLOG, Australian archives)

## License & data

Code: see [LICENSE]. Well data courtesy of the **Kansas Geological Survey** (public). Please cite KGS and this repo if you use the benchmark.

## Acknowledgments

Grew out of Ed Phillips' open petrophysics work. Built as an honest, open contribution to a problem the industry has quietly lived with for decades.
