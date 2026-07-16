"""The curve-extraction algorithm we iterate on together.

Emir's idea (v1): flood-fill from the left grid edge. Water flows right
until it hits strong curve ink and stops -> the fill edge IS the curve,
capturing peaks the old line-tracer missed.

extract(band, params) -> dict:
    curve   : x position per row (float, NaN where unknown)
    fill    : bool mask of the flooded region (for visualization)
    barrier : bool mask of the ink barrier (for visualization)
    stats   : {coverage, on_ink}
Everything here is meant to be edited live as we improve it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing


def _suppress_grid(dark: np.ndarray) -> np.ndarray:
    out = dark.astype(np.float32).copy()
    row_frac = out.mean(axis=1)
    out[row_frac > 0.6] *= 0.15
    col_frac = out.mean(axis=0)
    out[:, col_frac > 0.35] *= 0.15
    return out


def extract(band: np.ndarray, params: dict) -> dict:
    dark_thr = params.get("dark_threshold", 128)
    barrier_level = params.get("barrier_level", 0.35)
    gap_seal = int(params.get("gap_seal", 7))
    do_grid = params.get("grid_suppress", True)

    H, W = band.shape
    dark = (band < dark_thr).astype(float)
    ink = _suppress_grid(dark) if do_grid else dark
    barrier = ink > barrier_level

    # seal small vertical gaps (dashed / faded) so water doesn't leak
    if gap_seal > 1:
        barrier = binary_closing(barrier, structure=np.ones((gap_seal, 1)))

    # flood from the left edge through free (non-barrier) pixels
    free = ~barrier
    fill = np.zeros((H, W), bool)
    fill[:, 0] = free[:, 0]
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    prev = -1
    while fill.sum() != prev:
        prev = fill.sum()
        fill = binary_dilation(fill, structure=cross) & free

    # curve x per row = rightmost filled column
    curve = np.full(H, np.nan)
    for i in range(H):
        w = np.where(fill[i])[0]
        if len(w):
            curve[i] = w.max()

    valid = ~np.isnan(curve)
    coverage = float(valid.mean())
    on_ink = float(np.mean([
        barrier[i, min(W - 1, int(curve[i]) + 1)] or barrier[i, int(curve[i])]
        for i in range(H) if valid[i]])) if valid.any() else 0.0

    return dict(curve=curve, fill=fill, barrier=barrier,
                stats=dict(coverage=round(coverage, 3),
                           on_ink=round(on_ink, 3)))
