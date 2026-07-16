"""Render old vs momentum tracer on the same track, zoomed, for the eye.

python cloud/compare_tracers.py <well> [win_ft]
"""
from __future__ import annotations
import glob, os, sys, warnings
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extraction.layout import detect_layout, load_gray
from extraction.curves import extract_track_curves
from extraction.curves_momentum import extract_track_curves_momentum
warnings.filterwarnings("ignore"); Image.MAX_IMAGE_PIXELS = None
OUT = "data/tracer_compare"; os.makedirs(OUT, exist_ok=True)


def run(well, win_ft=200.0, track_idx=0):
    scan = glob.glob(f"data/pairs/{well}/scan_*")[0]
    lay = detect_layout(scan)
    gray = load_gray(scan)
    if not lay.tracks:
        print("no tracks"); return
    l, r = lay.tracks[track_idx]

    old = extract_track_curves(gray, (l, r), lay.log_top, lay.log_bottom, 2)
    new = extract_track_curves_momentum(gray, (l, r), lay.log_top,
                                        lay.log_bottom, 2)

    # zoom a window in the middle
    mid = (lay.log_top + lay.log_bottom) // 2
    # crude px/ft: assume ~15 px/ft if unknown
    win_px = min(2500, lay.log_bottom - lay.log_top)
    r0, r1 = max(lay.log_top, mid - win_px // 2), min(lay.log_bottom, mid + win_px // 2)

    def panel(curves, title, cols):
        crop = Image.fromarray(gray[r0:r1, l:r]).convert("RGB")
        d = ImageDraw.Draw(crop)
        span = (r - l - 12)
        for ci, cv in enumerate(curves):
            pts = [(6 + float(cv[row]) * span, row - r0) for row in range(r0, r1)]
            d.line(pts, fill=cols[ci], width=2)
        sc = 900 / crop.height
        crop = crop.resize((int(crop.width * sc), 900))
        return crop, title

    p1, t1 = panel(old, "OLD (Viterbi)", [(230, 30, 30), (30, 90, 240)])
    p2, t2 = panel(new, "NEW (momentum)", [(20, 170, 40), (240, 140, 0)])
    W = p1.width + p2.width + 30
    canvas = Image.new("RGB", (W, 940), (255, 255, 255))
    canvas.paste(p1, (0, 30)); canvas.paste(p2, (p1.width + 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"{well}  track {track_idx}   |   LEFT {t1}   RIGHT {t2}",
           fill=(0, 0, 0))
    out = f"{OUT}/{well}_t{track_idx}.png"
    canvas.save(out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 200.0,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0)
