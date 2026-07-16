"""Synthetic log-track generator for curve-segmentation pretraining.

We draw the curves ourselves, so we get PIXEL-PERFECT masks for free -
unlimited training data with no labeling. A UNet trained on these learns
to separate curve-ink from grid/text/noise, then transfers to real logs
(same trick that took the digit reader 4.5% -> 90%).

render() -> (image HxW uint8, curve_mask HxW uint8, per-curve x-paths)
"""

from __future__ import annotations

import numpy as np


def _smooth_curve(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth random curve path x(row) staying inside [0.05w, 0.95w].

    Real logs mix slow drift with occasional sharp kicks, so combine a
    low-frequency walk with sparse spikes.
    """
    steps = rng.normal(0, 1, h)
    # low-pass: cumulative sum then smooth
    walk = np.cumsum(steps)
    k = max(3, h // 30)
    kern = np.ones(k) / k
    walk = np.convolve(walk, kern, mode="same")
    # sparse sharp kicks (like a thin bed)
    for _ in range(rng.integers(0, 6)):
        c = rng.integers(0, h)
        wdt = rng.integers(2, 10)
        amp = rng.normal(0, 3)
        lo, hi = max(0, c - wdt), min(h, c + wdt)
        walk[lo:hi] += amp
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-6)
    lo = rng.uniform(0.05, 0.35)
    span = rng.uniform(0.3, 0.9 - lo)
    return (lo + walk * span) * w


def render(h: int = 320, w: int = 192, seed: int | None = None):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), rng.integers(238, 256), dtype=np.float32)

    # --- grid (always present, clearly visible so the model learns to
    #     reject it - grid is the #1 thing curve extraction must ignore) ---
    gc = rng.integers(120, 190)
    vstep = rng.integers(w // 14, w // 7)
    for x in range(0, w, vstep):
        img[:, x] = np.minimum(img[:, x], gc)
    hstep = rng.integers(h // 20, h // 8)
    for y in range(0, h, hstep):
        img[y, :] = np.minimum(img[y, :], gc)
    for x in range(0, w, max(1, vstep * 5)):  # heavier major lines
        img[:, max(0, x - 1):x + 1] = np.minimum(img[:, max(0, x - 1):x + 1], gc - 40)

    # --- curves ---
    n = rng.integers(1, 4)
    ink = rng.integers(0, 70)
    mask = np.zeros((h, w), dtype=np.uint8)
    paths = []
    for ci in range(n):
        x = _smooth_curve(h, w, rng)
        paths.append(x)
        lw = rng.integers(1, 3)
        dashed = rng.random() < 0.35
        for r in range(h):
            if dashed and (r // rng.integers(4, 9)) % 2 == 0:
                continue
            xi = int(round(x[r]))
            for dx in range(-lw, lw + 1):
                xx = xi + dx
                if 0 <= xx < w:
                    img[r, xx] = min(img[r, xx], ink + abs(dx) * 20)
                    mask[r, xx] = 1

    # --- artifacts ---
    if rng.random() < 0.6:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, rng.uniform(0.4, 1.1))
    if rng.random() < 0.6:
        img = img + rng.normal(0, rng.uniform(2, 12), img.shape)
    if rng.random() < 0.3:  # fading
        img = 255 - (255 - img) * rng.uniform(0.4, 0.85)

    return np.clip(img, 0, 255).astype(np.uint8), mask, paths


if __name__ == "__main__":
    from PIL import Image
    im, mask, _ = render(seed=1)
    Image.fromarray(im).save("data/synth_track.png")
    Image.fromarray(mask * 255).save("data/synth_mask.png")
    print("saved data/synth_track.png + data/synth_mask.png")
