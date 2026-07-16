"""Readable QC overlay: a zoomed depth window with traced curves drawn on
the scan at full detail, beside the aligned ground-truth curve.

python cloud/qc_zoom.py <well> [win_ft]
"""
from __future__ import annotations
import glob, os, sys, warnings
import numpy as np, pandas as pd, lasio
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves
warnings.filterwarnings("ignore"); Image.MAX_IMAGE_PIXELS = None
OUT = "data/qc_zoom"; os.makedirs(OUT, exist_ok=True)


def run(well: str, win_ft: float = 300.0):
    d = f"data/pairs/{well}"
    scan = glob.glob(f"{d}/scan_*")[0]
    layout = detect_layout(scan); cal = calibrate_depth(scan, layout)
    layout = reanchor_tracks(layout, cal.depth_band)
    gray = load_gray(scan)

    # a window in the middle of the log
    mid = (layout.log_top + layout.log_bottom) // 2
    win_px = int(win_ft / abs(cal.slope))
    r0, r1 = mid - win_px // 2, mid + win_px // 2
    r0, r1 = max(layout.log_top, r0), min(layout.log_bottom, r1)

    traces_by_track = {}
    for t, (l, rr) in enumerate(layout.tracks):
        traces_by_track[t] = extract_track_curves(
            gray, (l, rr), layout.log_top, layout.log_bottom, n_curves=2)

    crop = Image.fromarray(gray[r0:r1]).convert("RGB")
    dr = ImageDraw.Draw(crop)
    colors = [(230, 30, 30), (20, 90, 240), (20, 160, 50), (210, 120, 10)]
    for t, (l, rr) in enumerate(layout.tracks):
        for c, tr in enumerate(traces_by_track[t]):
            pts = [((l + 6) + float(tr[row]) * (rr - l - 12), row - r0)
                   for row in range(r0, r1)]
            dr.line(pts, fill=colors[(t * 2 + c) % 4], width=3)
    # shrink width only, keep vertical detail
    scale = 1100 / crop.height
    crop = crop.resize((int(crop.width * scale), 1100))
    d0, d1 = cal.depth_at(r0), cal.depth_at(r1)
    crop.save(f"{OUT}/{well}_{int(d0)}-{int(d1)}ft.png")
    print(f"{well}: {d0:.0f}-{d1:.0f} ft, RMS {cal.rms_residual_ft:.2f}, "
          f"tracks {len(layout.tracks)}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 300.0)
