"""Train the curve-segmentation UNet on synthetic log tracks.

Perfect masks from training.synth_tracks -> unlimited data, no labeling.
Runs locally on an 8 GB GPU. Val = fresh synthetic (held-out seeds) plus,
if present, a few hand-checked real crops later.

    python -m training.train_curveseg [--steps 4000]

Output: data/models/curveseg_unet.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.unet import UNet
from training.synth_tracks import render

MODEL_OUT = os.path.join("data", "models", "curveseg_unet.pt")
H, W = 320, 192


class SynthSeg(Dataset):
    def __init__(self, n: int, base_seed: int):
        self.n, self.base = n, base_seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, mask, _ = render(H, W, seed=self.base + i)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y


def dice_bce(logit, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tp = fp = fn = 0
    for x, y in dl:
        p = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
        tp += float((p * y).sum()); fp += float((p * (1 - y)).sum())
        fn += float(((1 - p) * y).sum())
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return prec, rec, f1


def main(steps: int = 4000, batch: int = 16, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    train_dl = DataLoader(SynthSeg(steps * batch, 0), batch_size=batch,
                          num_workers=6, persistent_workers=True)
    val_dl = DataLoader(SynthSeg(400, 10_000_000), batch_size=batch,
                        num_workers=4, persistent_workers=True)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    best = 0.0
    model.train()
    for i, (x, y) in enumerate(train_dl, 1):
        x, y = x.to(device), y.to(device)
        loss = dice_bce(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if i % 250 == 0 or i == steps:
            prec, rec, f1 = evaluate(model, val_dl, device)
            model.train()
            if f1 > best:
                best = f1
                torch.save(dict(state=model.state_dict(), h=H, w=W), MODEL_OUT)
            print(f"step {i:5d}  loss {float(loss):.3f}  "
                  f"val P {prec:.2f} R {rec:.2f} F1 {f1:.2f}  (best {best:.2f})",
                  flush=True)
        if i >= steps:
            break
    print(f"best synthetic val F1: {best:.2f} -> {MODEL_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    main(steps=a.steps)
