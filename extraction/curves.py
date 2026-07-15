"""Curve extraction: trace a log curve through a track.

Input: the track band (grid + curve ink). Output: the curve's x position
per row, normalized 0..1 across the track width.

Strategy:
1. suppress the grid — long horizontal runs and persistent vertical lines
   are grid, not curve
2. trace with dynamic programming (Viterbi): per row choose the x that
   minimizes (lack of ink) + (horizontal jump), which keeps the trace
   continuous through dashed segments, grid residue and small gaps
3. low-ink rows contribute no evidence; the path just coasts through them

Unit scaling to engineering values needs the header scale text (later);
for now the normalized trace is benchmarked against ground-truth LAS by
correlation, which is scale-free.
"""

from __future__ import annotations

import numpy as np

from .layout import DARK_THRESHOLD


def _suppress_grid(dark: np.ndarray) -> np.ndarray:
    """Return a copy with grid lines damped.

    - horizontal grid: rows whose dark fraction spans most of the track
    - vertical grid: columns dark in a large fraction of rows
    Damping (not zeroing) keeps curve pixels that overlap grid lines.
    """
    out = dark.astype(np.float32).copy()

    row_frac = out.mean(axis=1)
    hgrid = row_frac > 0.6
    out[hgrid] *= 0.15

    col_frac = out.mean(axis=0)
    vgrid = col_frac > 0.35
    out[:, vgrid] *= 0.15

    return out


def trace_curve(band: np.ndarray, step: int = 0, jump_penalty: float = 0.08,
                max_jump: int = 120) -> np.ndarray:
    # max_jump must cover near-horizontal curve slews (a GR kick can move
    # sideways >40 px per row); too small and the path can't follow them.
    # step 0 = adaptive: giant strips sample every 4 rows (still 2x finer
    # than the 0.5 ft output grid), normal strips every 2.
    if step == 0:
        step = 4 if band.shape[0] > 30000 else 2
    """Trace the dominant curve through a grayscale track band.

    Returns x position (float, band coordinates) per sampled row; rows are
    sampled every `step` px. Positions for skipped rows are interpolated.
    """
    dark = (band < DARK_THRESHOLD)
    ink = _suppress_grid(dark)

    rows = np.arange(0, band.shape[0], step)
    n, w = len(rows), band.shape[1]

    # evidence: more ink -> lower cost. Blur slightly so near-misses count.
    evidence = ink[rows]
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
    for i in range(evidence.shape[0]):
        evidence[i] = np.convolve(evidence[i], kernel, mode="same")
    cost_local = 1.0 - np.clip(evidence, 0, 1)

    # Viterbi with banded transitions
    offsets = np.arange(-max_jump, max_jump + 1)
    trans = jump_penalty * (np.abs(offsets) / max_jump) ** 2

    total = cost_local[0].copy()
    back = np.zeros((n, w), dtype=np.int32)
    for i in range(1, n):
        # for each x, best predecessor within the jump band
        stacked = np.full((len(offsets), w), np.inf, dtype=np.float32)
        for k, off in enumerate(offsets):
            lo, hi = max(0, -off), min(w, w - off)
            stacked[k, lo:hi] = total[lo + off:hi + off] + trans[k]
        k_best = np.argmin(stacked, axis=0)
        total = stacked[k_best, np.arange(w)] + cost_local[i]
        back[i] = np.arange(w) + offsets[k_best]

    # backtrack
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmin(total))
    for i in range(n - 2, -1, -1):
        path[i] = back[i + 1][path[i + 1]]

    # interpolate to every row
    full = np.interp(np.arange(band.shape[0]), rows, path.astype(float))
    return full


def extract_track_curve(gray: np.ndarray, track: tuple[int, int],
                        log_top: int, log_bottom: int,
                        margin: int = 6) -> np.ndarray:
    """Trace the dominant curve in one track; values normalized 0..1."""
    return extract_track_curves(gray, track, log_top, log_bottom,
                                n_curves=1, margin=margin)[0]


def extract_track_curves(gray: np.ndarray, track: tuple[int, int],
                         log_top: int, log_bottom: int, n_curves: int = 2,
                         margin: int = 6, erase_px: int = 9) -> list[np.ndarray]:
    """Trace up to n_curves curves in one track (e.g. GR + caliper).

    Iteratively: trace the cheapest path, erase its ink, trace again.
    Which trace is which curve is decided downstream (variability, line
    style, or correlation against known curves).
    Returns a list of arrays normalized 0..1 across the track width.
    """
    left, right = track
    band = gray[log_top:log_bottom, left + margin:right - margin].copy()
    w = band.shape[1]

    curves = []
    for _ in range(n_curves):
        x = trace_curve(band)
        curves.append(x / max(1.0, w - 1))
        # erase the traced path so the next pass finds the next curve
        rows = np.arange(band.shape[0])
        for dx in range(-erase_px, erase_px + 1):
            xs = np.clip(x.astype(int) + dx, 0, w - 1)
            band[rows, xs] = 255
    return curves
