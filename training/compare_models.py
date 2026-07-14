"""Evaluate two model checkpoints on the SAME held-out well split.

Val accuracy is only comparable on an identical exam; cycle retraining
changes the manifest and therefore the split, so cross-cycle 'best'
numbers cannot be compared directly. This evaluates any checkpoints on
the current seed-7 by-well split.

Usage:
    python -m training.compare_models ckpt_a.pt ckpt_b.pt
"""

from __future__ import annotations

import csv
import random
import sys

import torch
from torch.utils.data import DataLoader

from training.train_digits import (CRNN, LabelCrops, collate, _evaluate,
                                   MANIFEST)


def main(paths: list[str]):
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    wells = sorted({r["well"] for r in rows})
    random.Random(7).shuffle(wells)
    val_wells = set(wells[:max(1, len(wells) // 5)])
    val_rows = [r for r in rows if r["well"] in val_wells]
    print(f"fixed exam: {len(val_rows)} crops from {len(val_wells)} "
          f"held-out wells")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dl = DataLoader(LabelCrops(val_rows, False), batch_size=64,
                        shuffle=False, collate_fn=collate, num_workers=8)

    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        model = CRNN()
        model.load_state_dict(ckpt["state"])
        model = model.to(device).eval()
        acc = _evaluate(model, val_dl, device)
        print(f"{path}: exact-match {acc:.1%}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
