"""Synthetic depth-label generator for CRNN pretraining.

Renders random plausible depth values (round numbers, 2-5 digits) in
print-like fonts with the artifacts real crops show: border lines, grid
ticks, blur, noise, skew. Unlimited data for pretraining; the real
harvested crops then fine-tune.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = ["arialbd.ttf", "arial.ttf", "cour.ttf", "courbd.ttf",
         "times.ttf", "timesbd.ttf", "consola.ttf", "calibrib.ttf"]


def _font(size: int):
    for _ in range(4):
        try:
            return ImageFont.truetype(random.choice(FONTS), size)
        except OSError:
            continue
    return ImageFont.load_default()


def random_value() -> int:
    step = random.choice([25, 50, 50, 100, 100])
    return random.randrange(50, 12000, step)


def render(value: int | None = None) -> tuple[np.ndarray, str]:
    """One synthetic crop (grayscale uint8) + its text label."""
    value = random_value() if value is None else value
    text = str(value)

    size = random.randint(22, 54)
    font = _font(size)
    pad_x, pad_y = random.randint(8, 40), random.randint(6, 24)

    tmp = Image.new("L", (10, 10))
    x0, y0, x1, y1 = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y

    im = Image.new("L", (w, h), random.randint(225, 255))
    d = ImageDraw.Draw(im)
    ink = random.randint(0, 60)
    d.text((pad_x - x0, pad_y - y0), text, font=font, fill=ink)

    # artifacts real crops show
    if random.random() < 0.5:   # vertical border line(s) at the edges
        for x in random.sample([1, 2, w - 2, w - 3], k=random.randint(1, 2)):
            d.line([x, 0, x, h], fill=random.randint(0, 80),
                   width=random.randint(1, 3))
    if random.random() < 0.4:   # grid tick stubs
        for _ in range(random.randint(1, 5)):
            y = random.randint(0, h - 1)
            side = random.choice([0, w - random.randint(4, 10)])
            d.line([side, y, side + random.randint(4, 10), y],
                   fill=random.randint(0, 90), width=1)
    if random.random() < 0.35:  # slight rotation/skew
        im = im.rotate(random.uniform(-2.5, 2.5), expand=False,
                       fillcolor=255)
    if random.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))

    arr = np.asarray(im, dtype=np.float32)
    if random.random() < 0.5:   # scanner noise
        arr = arr + np.random.normal(0, random.uniform(2, 10), arr.shape)
    if random.random() < 0.3:   # fade (old paper)
        arr = 255 - (255 - arr) * random.uniform(0.45, 0.9)
    return np.clip(arr, 0, 255).astype(np.uint8), text
