# Contributing to LogLift

The headline goal is **better curve extraction.** Depth reading works; curves are the open problem. This guide shows you how to plug in your idea, measure it against ground truth, and put a number on the board.

## The challenge, precisely

Given a scanned log track (a grayscale image), output the **x-position of each curve per depth**, so it can be scaled to engineering units and written to LAS. The enemies:

1. **Grid vs curve** — a sharp rightward peak looks locally identical to a horizontal grid line.
2. **Overlapping curves** — GR and caliper cross constantly; which trace is which?
3. **Chaotic zones** — washed-out intervals where the pen scribbles and there's barely a single "value."
4. **Dashed / faded ink**, 90 years of formats, rotated labels, composite strips.

The best published method ([VeerNet, 2023](https://www.mdpi.com/2313-433X/9/7/136)) reaches ~0.62 gamma-ray correlation. That's the bar.

## Where the baseline lives

- `extraction/curves.py` — the current Viterbi line-tracer (baseline)
- `playground/` — a single-log visual sandbox (`python -m playground.server`, port 3000) to iterate on an algorithm and *see* the result on a real track
- `training/synth_tracks.py`, `models/unet.py`, `training/train_curveseg.py` — scaffolding for a learned (UNet segmentation) approach, including a synthetic-track generator for pretraining
- `training/curve_labels.py` — self-labeling: project ground-truth LAS curves back onto scans to auto-generate training data

## How to measure your improvement

The benchmark is the scorer. It runs your pipeline on every well and correlates each extracted curve against the ground-truth LAS, with a small depth-lag search.

```bash
python -m pipeline.download --limit 500   # get a test set
python benchmark.py --workers 4           # writes data/benchmark.csv
```

Report **median best-curve correlation** and **% of wells with a strong match (|r| ≥ 0.7)**, plus depth RMS (should stay ~0.8 ft). Add your numbers to [LEADERBOARD.md](LEADERBOARD.md) with a one-line description and a link to your branch/PR.

### Honest caveat on the metric

Some ground-truth LAS files are a *different logging run* than the scan, so perfect tracing can't correlate. For a fair comparison, prefer the high-coverage subset (curves that already correlate ≥ 0.5 with the baseline are known scan↔LAS matches). Improving the metric definition is itself a welcome contribution.

## Ground rules

- **Don't fabricate.** The pipeline refuses when it can't read something rather than inventing data. Keep that. A confident wrong answer is worse than an honest "needs review."
- **Reproducible.** If you add a model, include training code and a way to regenerate weights.
- **Regression-check depth.** Don't break the depth pipeline (0.82 ft) while improving curves.

## Getting the data

Everything downloads free from the Kansas Geological Survey via `pipeline/` — no account, no key. The full set is 22,490 paired wells; start with a few hundred.

## Ideas that need someone

- UNet curve segmentation trained on synthetic + self-labeled tracks
- Peak-vs-grid disambiguation using full-width detection or connectivity
- Instance separation for overlapping curves (line style, momentum tracking)
- A confidence score per curve so the app knows what to flag for human review
- International logs (metric scales, other-language headers)

Open an issue to discuss an approach before a big PR. Small, measured improvements are very welcome.
