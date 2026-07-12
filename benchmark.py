"""End-to-end benchmark: scan -> layout -> depth -> curves vs LAS truth.

For every downloaded pair:
1. detect layout, calibrate depth
2. trace 2 curves per track
3. correlate each trace against every LAS curve (small depth-lag search)
4. record the best |r| match per trace

Output: data/benchmark.csv with one row per traced curve.

Usage:
    python benchmark.py
"""

from __future__ import annotations

import glob
import os
import warnings

import lasio
import numpy as np
import pandas as pd

from extraction.layout import detect_layout, load_gray
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves

warnings.filterwarnings("ignore")


def best_las_match(trace: np.ndarray, scan_depths: np.ndarray,
                   las) -> tuple[str, float, float]:
    """(mnemonic, r, lag_ft) of the LAS curve best matching the trace."""
    depths = np.asarray(las.index, dtype=float)
    best = ("", 0.0, 0.0)
    for curve in las.curves[1:]:
        vals = np.asarray(las[curve.mnemonic], dtype=float)
        ok = ~np.isnan(vals)
        if ok.sum() < 100:
            continue
        for lag in np.arange(-10, 10.5, 2.5):
            gt = np.interp(scan_depths + lag, depths[ok], vals[ok],
                           left=np.nan, right=np.nan)
            v = ~np.isnan(gt)
            if v.sum() < 500:
                continue
            r = float(np.corrcoef(trace[v], gt[v])[0, 1])
            if abs(r) > abs(best[1]):
                best = (curve.mnemonic, r, float(lag))
    return best


def run(pairs_dir: str = "data/pairs", out_csv: str = "data/benchmark.csv"):
    rows = []
    for well_dir in sorted(glob.glob(os.path.join(pairs_dir, "*"))):
        well = os.path.basename(well_dir)
        scans = glob.glob(os.path.join(well_dir, "scan_*"))
        las_files = glob.glob(os.path.join(well_dir, "las_*"))
        if not scans or not las_files:
            continue
        scan = scans[0]
        try:
            layout = detect_layout(scan)
            cal = calibrate_depth(scan, layout)
        except Exception as e:
            rows.append(dict(well=well, stage="depth",
                             error=f"{type(e).__name__}: {str(e)[:60]}"))
            continue

        try:
            las = lasio.read(las_files[0])
        except Exception as e:
            rows.append(dict(well=well, stage="las",
                             error=f"{type(e).__name__}: {str(e)[:60]}"))
            continue

        gray = load_gray(scan)
        scan_rows = np.arange(layout.log_top, layout.log_bottom)
        scan_depths = cal.slope * scan_rows + cal.intercept

        for t_idx, track in enumerate(layout.tracks):
            try:
                traces = extract_track_curves(
                    gray, track, layout.log_top, layout.log_bottom, n_curves=2)
            except Exception as e:
                rows.append(dict(well=well, stage=f"trace_t{t_idx}",
                                 error=f"{type(e).__name__}: {str(e)[:60]}"))
                continue
            for c_idx, trace in enumerate(traces):
                mnem, r, lag = best_las_match(trace, scan_depths, las)
                rows.append(dict(
                    well=well, stage="ok", track=t_idx, curve=c_idx,
                    trace_std=round(float(np.std(trace)), 3),
                    match=mnem, r=round(r, 3), lag_ft=lag,
                    rms_ft=round(cal.rms_residual_ft, 2),
                ))
        print(f"{well}: done")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    ok = df[df.stage == "ok"]
    if len(ok):
        strong = ok[ok.r.abs() >= 0.7]
        print(f"\ntraces benchmarked: {len(ok)} across {ok.well.nunique()} wells")
        print(f"strong matches (|r|>=0.7): {len(strong)} "
              f"({len(strong)/len(ok):.0%}) on {strong.well.nunique()} wells")
        print("match mnemonics:", strong.match.value_counts().to_dict())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    run()
