#!/bin/bash
# One self-labeling cycle: harvest crops with the current model, retrain.
# Safe to re-run; harvest skips wells already in the manifest.
#
# WORKERS: ~4 per GPU works well (each holds ~1 GB VRAM);
# on a 4-GPU/128-core box try WORKERS=16.
set -euo pipefail

WORKERS="${WORKERS:-8}"

echo "== harvest (workers=$WORKERS)"
python -m training.harvest_labels --workers "$WORKERS"

echo "== crop count"
ls data/label_crops/*.png | wc -l

echo "== retrain"
python -m training.train_digits

echo "== cycle done; re-run to keep ratcheting, or run the benchmark:"
echo "   python benchmark.py --workers $WORKERS"
