"""Train a small CRNN to read depth labels (digit strings) from crops.

Architecture: CNN feature extractor -> BiLSTM -> CTC. The standard shape
for variable-length text; tiny enough to train in minutes on a laptop GPU.

Data: crops harvested by training.harvest_labels (real, self-labeled) plus
on-the-fly augmentation (shift/rotate/noise/contrast). Split train/val BY
WELL so validation measures generalization to unseen wells, not unseen
crops of the same print.

Usage:
    python -m training.train_digits [--epochs 60]

Output: data/models/digit_crnn.pt  (weights + charset metadata)
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MANIFEST = os.path.join("data", "label_crops", "manifest.csv")
MODEL_OUT = os.path.join("data", "models", "digit_crnn.pt")

CHARS = "0123456789"
BLANK = len(CHARS)  # CTC blank index
IMG_H = 32
MAX_W = 256


def encode(text: str) -> list[int]:
    return [CHARS.index(c) for c in text]


class LabelCrops(Dataset):
    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        im = Image.open(path).convert("L")
        # normalize height, keep aspect
        w = max(8, int(im.width * IMG_H / im.height))
        im = im.resize((min(w, MAX_W), IMG_H), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    def __getitem__(self, i):
        row = self.rows[i]
        arr = self._load(row["path"])
        if self.augment:
            if random.random() < 0.5:  # contrast jitter
                arr = np.clip((arr - 0.5) * random.uniform(0.6, 1.6) + 0.5, 0, 1)
            if random.random() < 0.3:  # noise
                arr = np.clip(arr + np.random.normal(0, 0.05, arr.shape), 0, 1)
            if random.random() < 0.3:  # small vertical shift
                arr = np.roll(arr, random.randint(-2, 2), axis=0)
        target = encode(str(int(row["value"])))
        return torch.from_numpy(arr)[None], torch.tensor(target)


def collate(batch):
    widths = [b[0].shape[-1] for b in batch]
    w = max(widths)
    imgs = torch.ones(len(batch), 1, IMG_H, w)
    targets, target_lens = [], []
    for i, (im, tgt) in enumerate(batch):
        imgs[i, :, :, :im.shape[-1]] = im
        targets.append(tgt)
        target_lens.append(len(tgt))
    return imgs, torch.cat(targets), torch.tensor(target_lens), torch.tensor(widths)


class CRNN(nn.Module):
    def __init__(self, n_classes=len(CHARS) + 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # H16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # H8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                                          # H4
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((4, 1)),                                          # H1
        )
        self.rnn = nn.LSTM(128, 96, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(192, n_classes)

    def forward(self, x):                       # x: B,1,32,W
        f = self.cnn(x)                         # B,128,1,W/4
        f = f.squeeze(2).permute(0, 2, 1)       # B,W/4,128
        f, _ = self.rnn(f)
        return self.fc(f)                       # B,W/4,classes


def ctc_decode(logits: torch.Tensor) -> str:
    ids = logits.argmax(-1).tolist()
    out, prev = [], BLANK
    for i in ids:
        if i != prev and i != BLANK:
            out.append(CHARS[i])
        prev = i
    return "".join(out)


def main(epochs: int = 60, batch: int = 32, lr: float = 1e-3):
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    wells = sorted({r["well"] for r in rows})
    random.Random(7).shuffle(wells)
    val_wells = set(wells[:max(1, len(wells) // 5)])
    train_rows = [r for r in rows if r["well"] not in val_wells]
    val_rows = [r for r in rows if r["well"] in val_wells]
    print(f"crops: {len(train_rows)} train / {len(val_rows)} val "
          f"({len(wells)} wells, {len(val_wells)} held out)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    train_dl = DataLoader(LabelCrops(train_rows, True), batch_size=batch,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(LabelCrops(val_rows, False), batch_size=batch,
                        shuffle=False, collate_fn=collate)

    best_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for imgs, targets, tlens, widths in train_dl:
            imgs = imgs.to(device)
            logits = model(imgs)                       # B,T,C
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            in_lens = torch.full((imgs.shape[0],), logits.shape[1],
                                 dtype=torch.long)
            loss = ctc(log_probs, targets, in_lens, tlens)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)

        model.eval()
        correct = n = 0
        with torch.no_grad():
            for imgs, targets, tlens, widths in val_dl:
                logits = model(imgs.to(device))
                pos = 0
                for bi in range(imgs.shape[0]):
                    truth = "".join(CHARS[t] for t in
                                    targets[pos:pos + tlens[bi]].tolist())
                    pos += int(tlens[bi])
                    correct += ctc_decode(logits[bi]) == truth
                    n += 1
        acc = correct / max(1, n)
        if acc > best_acc:
            best_acc = acc
            torch.save(dict(state=model.state_dict(), chars=CHARS,
                            img_h=IMG_H), MODEL_OUT)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {total/len(train_dl):.3f}  "
                  f"val exact-match {acc:.1%}  (best {best_acc:.1%})",
                  flush=True)

    print(f"best val exact-match: {best_acc:.1%} -> {MODEL_OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    main(epochs=args.epochs, batch=args.batch)
