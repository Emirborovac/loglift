"""LogLift converter: scanned well log (TIFF) in -> LAS file out.

Usage:
    python convert.py path/to/scan.tif [-o out.las] [--well NAME] [--api API]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves
from extraction.header import read_header
from export.las_writer import write_las

warnings.filterwarnings("ignore")


def _assign_curves(traces: list, scales: list) -> list:
    """Pair traced curves with header-identified curves (v0 heuristics).

    - GR + CALI in one track: GR is the variable trace, CALI the flat one
    - otherwise scales (top-down) map to traces in traced order
    Traces without a scale stay normalized; names get '?' when the pairing
    is a guess so users know to verify.
    """
    out = []
    mnems = {s.mnemonic for s in scales}
    stds = [float(np.std(t)) for t in traces]

    if {"GR", "CALI"} <= mnems and len(traces) >= 2:
        gr_i = int(np.argmax(stds))
        by_m = {s.mnemonic: s for s in scales}
        pairs = [(traces[gr_i], by_m["GR"]),
                 (traces[1 - gr_i], by_m["CALI"])]
    else:
        pairs = list(zip(traces, scales))

    used = {id(t) for t, _ in pairs}
    for trace, scale in pairs:
        sure = {"GR", "CALI"} <= mnems or len(scales) == len(traces) == 1
        name = scale.mnemonic if sure else scale.mnemonic + "?"
        if scale.left_value is not None and scale.right_value is not None:
            values = scale.left_value + np.asarray(trace) * (
                scale.right_value - scale.left_value)
            out.append(dict(name=name, unit=scale.unit, values=values,
                            descr="scaled from header scale line"))
        else:
            out.append(dict(name=name, unit="norm", values=trace,
                            descr="header named it; no scale endpoints "
                                  "printed - normalized 0..1"))
    for trace in traces:
        if id(trace) not in used:
            out.append(trace)
    return out


def convert(scan_path: str, out_path: str | None = None,
            well_name: str = "", api: str = "",
            curves_per_track: int = 2) -> str:
    out_path = out_path or os.path.splitext(scan_path)[0] + "_loglift.las"

    print(f"[1/4] layout: {os.path.basename(scan_path)}")
    layout = detect_layout(scan_path)
    if not layout.tracks:
        raise SystemExit("no curve tracks detected - cannot convert")
    print(f"      log rows {layout.log_top}-{layout.log_bottom}, "
          f"{len(layout.tracks)} tracks, depth column {layout.depth_col}")

    print("[2/4] depth calibration (OCR)")
    cal = calibrate_depth(scan_path, layout)
    layout = reanchor_tracks(layout, cal.depth_band)
    d0, d1 = cal.depth_at(layout.log_top), cal.depth_at(layout.log_bottom)
    print(f"      {cal.n_inliers} labels, RMS {cal.rms_residual_ft:.2f} ft, "
          f"depth {d0:.0f}-{d1:.0f} ft, depth column {layout.depth_col}")

    print("[3/4] curve tracing + header identification")
    try:
        scales = read_header(scan_path, layout)
    except Exception:
        scales = []
    for s in scales:
        print(f"      header: track {s.track} {s.mnemonic} [{s.unit}] "
              f"{s.left_value} -> {s.right_value}")

    gray = load_gray(scan_path)
    track_curves = {}
    for t_idx, track in enumerate(layout.tracks):
        traces = extract_track_curves(
            gray, track, layout.log_top, layout.log_bottom,
            n_curves=curves_per_track)
        t_scales = [s for s in scales if s.track == t_idx]
        track_curves[t_idx] = _assign_curves(traces, t_scales)
        print(f"      track {t_idx}: {len(traces)} curves traced")

    print("[4/4] writing LAS")
    write_las(out_path, layout, cal, track_curves,
              well_name=well_name, api=api,
              source_scan=os.path.basename(scan_path))
    print(f"done -> {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert a scanned well log to LAS")
    p.add_argument("scan", help="path to the scan (TIFF)")
    p.add_argument("-o", "--out", default=None, help="output LAS path")
    p.add_argument("--well", default="", help="well name for the LAS header")
    p.add_argument("--api", default="", help="API number for the LAS header")
    args = p.parse_args()
    convert(args.scan, args.out, well_name=args.well, api=args.api)
