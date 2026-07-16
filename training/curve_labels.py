"""Self-labeling for curves: paint the true LAS curve path onto the scan.

For a well where the scan and LAS genuinely align, we can generate a
pixel-perfect training label for the curve tracer with zero manual work:

1. depth calibration maps scan row <-> depth
2. resample the ground-truth LAS curve onto scan rows
3. recover the horizontal scale (value -> x) by a linear fit against the
   existing (imperfect) extracted trace -- the old trace is only used to
   find the SCALE; the LAS gives the clean SHAPE
4. project: x_label(row) = a * las_value(row) + b   -> the true curve path

Output is a normalized 0..1 path per row: the label a learned tracer
trains on. This module verifies the idea; the training pipeline consumes it.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import lasio

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves


def project_curve(well: str, track_idx: int, curve_idx: int,
                  mnemonic: str, lag_ft: float = 0.0):
    """Return (rows, label_norm, trace_norm, quality) for one well/curve."""
    scan = glob.glob(f"data/pairs/{well}/scan_*")[0]
    las = lasio.read(glob.glob(f"data/pairs/{well}/las_*")[0])

    layout = detect_layout(scan)
    cal = calibrate_depth(scan, layout)
    layout = reanchor_tracks(layout, cal.depth_band)
    gray = load_gray(scan)

    rows = np.arange(layout.log_top, layout.log_bottom)
    depths = cal.slope * rows + cal.intercept

    # ground-truth curve resampled onto scan rows
    las_d = pd.to_numeric(np.asarray(las.index), errors="coerce")
    las_v = pd.to_numeric(np.asarray(las[mnemonic]), errors="coerce")
    ok = ~(np.isnan(las_d) | np.isnan(las_v))
    gt = np.interp(depths + lag_ft, las_d[ok], las_v[ok],
                   left=np.nan, right=np.nan)

    # existing trace (only to recover the value->x scale)
    traces = extract_track_curves(gray, layout.tracks[track_idx],
                                  layout.log_top, layout.log_bottom, n_curves=2)
    trace = traces[curve_idx]

    v = ~np.isnan(gt)
    if v.sum() < 500:
        return rows, None, trace, dict(ok=False, reason="too few overlap")
    # linear fit: trace_norm ~= a*gt + b
    a, b = np.polyfit(gt[v], trace[v], 1)
    label = a * gt + b                      # clean projected path (normalized)
    label = np.clip(label, 0, 1)

    # quality: how well the projected label agrees with the ink under it
    resid = np.nanstd(trace[v] - label[v])
    return rows, label, trace, dict(ok=True, a=float(a), b=float(b),
                                    resid=float(resid),
                                    corr=float(np.corrcoef(trace[v], gt[v])[0, 1]))
