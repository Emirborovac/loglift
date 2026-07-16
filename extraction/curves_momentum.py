"""Momentum curve tracker (Version A: pure geometry, no neural network).

Idea (Emir's): don't trace one curve down the whole page. Walk down in
small blocks, and carry each curve's DIRECTION as memory. When a block
offers two candidates (a crossing), pick the one that continues the
previous trajectory. Curves are smooth, so momentum resolves crossings.

This is a small multi-object tracker running top-to-bottom:
- each row, find ink "blobs" (a curve crossing = a short cluster of dark px)
- predict each tracked curve's next x from its velocity, match to nearest
  blob, and update velocity (smoothed) = momentum
- at a crossing two curves may share one blob; both coast on prediction
  and separate again after

Returns one x-array per tracked curve, normalized 0..1 across the track.
"""

from __future__ import annotations

import numpy as np

from .layout import DARK_THRESHOLD
from .curves import _suppress_grid


def _row_blobs(ink_row: np.ndarray, min_gap: int = 3) -> list[float]:
    """Centers of contiguous ink runs in one row."""
    xs = np.where(ink_row > 0.3)[0]
    if len(xs) == 0:
        return []
    blobs, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > min_gap:
            blobs.append((start + prev) / 2.0)
            start = x
        prev = x
    blobs.append((start + prev) / 2.0)
    return blobs


def track_curves(band: np.ndarray, n_curves: int = 2,
                 gate: float = 18.0, vel_smooth: float = 0.6,
                 max_coast: int = 60) -> list[np.ndarray]:
    """Track up to n_curves down the band using directional memory."""
    dark = (band < DARK_THRESHOLD)
    ink = _suppress_grid(dark)
    h, w = band.shape

    # seed: first row (scanning down) that has enough distinct blobs
    seed_row, seed_blobs = 0, []
    for r in range(h):
        b = _row_blobs(ink[r])
        if len(b) >= n_curves:
            seed_row, seed_blobs = r, sorted(b)[:n_curves]
            break
    if not seed_blobs:
        # fall back: single strongest column
        seed_blobs = [float(np.argmax(ink.sum(axis=0)))]

    # one tracker per curve: position x, velocity vx, coast counter, path
    tracks = [dict(x=x, vx=0.0, coast=0, path={}) for x in seed_blobs]
    for t in tracks:
        t["path"][seed_row] = t["x"]

    def step(rows):
        for r in rows:
            blobs = _row_blobs(ink[r])
            used = []
            for t in tracks:
                x_pred = t["x"] + t["vx"]
                # nearest blob to the predicted position within the gate
                best, bestd = None, gate + 1
                for b in blobs:
                    d = abs(b - x_pred)
                    if d < bestd:
                        best, bestd = b, d
                if best is not None:
                    # momentum update; a blob may serve two tracks (crossing)
                    new_vx = best - t["x"]
                    t["vx"] = vel_smooth * t["vx"] + (1 - vel_smooth) * new_vx
                    t["x"] = best
                    t["coast"] = 0
                    used.append(best)
                else:
                    # gap or crossing: coast on predicted trajectory
                    t["x"] = min(w - 1, max(0, x_pred))
                    t["coast"] += 1
                    if t["coast"] > max_coast:
                        t["vx"] *= 0.5  # trajectory stale, damp it
                t["path"][r] = t["x"]

    step(range(seed_row + 1, h))          # downward
    step(range(seed_row - 1, -1, -1))     # upward to fill the top

    out = []
    for t in tracks:
        arr = np.array([t["path"].get(r, np.nan) for r in range(h)])
        # fill any gaps
        idx = np.arange(h)
        good = ~np.isnan(arr)
        if good.any():
            arr = np.interp(idx, idx[good], arr[good])
        out.append(arr / max(1.0, w - 1))
    return out


def extract_track_curves_momentum(gray, track, log_top, log_bottom,
                                  n_curves=2, margin=6):
    left, right = track
    band = gray[log_top:log_bottom, left + margin:right - margin]
    return track_curves(band, n_curves=n_curves)
