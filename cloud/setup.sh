#!/bin/bash
# LogLift cloud bootstrap for a fresh Vast.ai instance (pytorch template).
#
# Usage:
#   git clone https://github.com/Emirborovac/loglift.git && cd loglift
#   bash cloud/setup.sh          # deps + dataset download (~30-60 min)
#   bash cloud/cycle.sh          # one harvest -> train cycle
#
# Re-run cycle.sh until the harvest adds few new wells (converged), then:
#   python benchmark.py --workers $WORKERS
set -euo pipefail

echo "== [1/3] python deps"
pip install -q -r requirements.txt

echo "== [2/3] build pair manifests from KGS indexes"
python -m pipeline.indexes
python -m pipeline.pairs

echo "== [3/3] download the full paired dataset (resumable)"
python -m pipeline.download --limit 14200 --workers 24

echo "setup complete:"
ls data/pairs | wc -l
df -h .
