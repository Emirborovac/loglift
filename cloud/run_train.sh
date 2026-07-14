#!/bin/bash
# Detached training launcher (avoids ssh quoting hazards).
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
exec python -u -m training.train_digits
