# LogLift Leaderboard — curve extraction

Scores on the 13,112-well benchmark (`python benchmark.py`). The open goal
is to raise curve-extraction quality past the baseline and toward / beyond
the published state of the art.

## Curve extraction

| Method | Median best-curve r | Strong (\|r\|≥0.7) | Depth RMS | Notes | By |
|---|---|---|---|---|---|
| VeerNet (2023, reference SOTA) | ~0.62 (GR) | — | — | UNet+attention, 10k images | published |
| **LogLift baseline** (Viterbi tracer) | **0.51** | **25%** | 0.82 ft | classical, no ML | core |
| *your method here* | | | | | you |

## Depth digitization (mostly solved — keep it from regressing)

| Method | Auto-convert rate | Depth RMS | Depth reader | By |
|---|---|---|---|---|
| **LogLift baseline** | **77.7%** | **0.82 ft** | CRNN ~90% held-out | core |

## How to submit

1. Run `python benchmark.py --workers N` on a documented test set (state how many wells).
2. Report median best-curve r, strong-match %, and depth RMS.
3. Open a PR adding a row here with a link to your branch and a one-line method description.
4. Include reproduction steps (and training code / weights if it's a model).

Honest, reproducible numbers only. See [CONTRIBUTING.md](CONTRIBUTING.md) for the metric caveats.
