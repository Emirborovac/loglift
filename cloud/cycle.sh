#!/bin/bash
# One self-labeling cycle: harvest crops with the current model, retrain.
# Safe to re-run; harvest skips wells already in the manifest.
#
# WORKERS: ~4 per GPU works well (each holds ~1 GB VRAM);
# on a 4-GPU/128-core box try WORKERS=16.
set -euo pipefail

WORKERS="${WORKERS:-8}"
export N_GPUS="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MIN_WELLS="${MIN_WELLS:-200}"

echo "== harvest (workers=$WORKERS, gpus=$N_GPUS)"
python -m training.harvest_labels --workers "$WORKERS"

echo "== crop count"
find data/label_crops -name '*.png' | wc -l

# guard: never retrain (and overwrite the model) on a failed harvest
WELLS=$(tail -n +2 data/label_crops/manifest.csv | cut -d, -f3 | sort -u | wc -l)
echo "== wells in manifest: $WELLS"
if [ "$WELLS" -lt "$MIN_WELLS" ]; then
    echo "harvest yielded only $WELLS wells (<$MIN_WELLS) - refusing to retrain"
    exit 1
fi

echo "== retrain"
python -m training.train_digits

echo "== cycle done; re-run to keep ratcheting, or run the benchmark:"
echo "   python benchmark.py --workers $WORKERS"
