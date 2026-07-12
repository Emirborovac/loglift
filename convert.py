"""LogLift converter: scanned well log (TIFF) in -> LAS file out.

Usage:
    python convert.py path/to/scan.tif [-o out.las] [--well NAME] [--api API]
"""

from __future__ import annotations

import argparse
import os
import warnings

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves
from export.las_writer import write_las

warnings.filterwarnings("ignore")


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

    print("[3/4] curve tracing")
    gray = load_gray(scan_path)
    track_curves = {}
    for t_idx, track in enumerate(layout.tracks):
        track_curves[t_idx] = extract_track_curves(
            gray, track, layout.log_top, layout.log_bottom,
            n_curves=curves_per_track)
        print(f"      track {t_idx}: {curves_per_track} curves traced")

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
