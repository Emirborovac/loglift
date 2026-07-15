"""Render QC overlays for a list of wells: traced curves on the scan +
the ground-truth LAS curve beside it, for visual review.

Usage:
    python cloud/make_overlays.py <well_id> <well_id> ...
Outputs data/qc_overlays/<well>.png
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lasio
from PIL import Image, ImageDraw

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

OUT = "data/qc_overlays"


def overlay(well: str):
    d = f"data/pairs/{well}"
    scans = glob.glob(f"{d}/scan_*")
    lasf = glob.glob(f"{d}/las_*")
    if not scans or not lasf:
        return
    scan = scans[0]
    layout = detect_layout(scan)
    cal = calibrate_depth(scan, layout)
    layout = reanchor_tracks(layout, cal.depth_band)
    gray = load_gray(scan)

    # traced curves overlaid on a downsampled scan
    im = Image.fromarray(gray).convert("RGB")
    scale = 700 / im.width
    small = im.resize((700, int(im.height * scale)))
    dr = ImageDraw.Draw(small)
    colors = [(230, 40, 40), (30, 110, 230), (30, 160, 60), (200, 120, 20)]
    for t_idx, (l, r) in enumerate(layout.tracks):
        span = (r - l - 12) * scale
        x0 = (l + 6) * scale
        traces = extract_track_curves(gray, (l, r), layout.log_top,
                                      layout.log_bottom, n_curves=2)
        for c_idx, tr in enumerate(traces):
            pts = [(x0 + float(tr[row]) * span, (layout.log_top + row) * scale)
                   for row in range(0, len(tr), 8)]
            if len(pts) > 1:
                dr.line(pts, fill=colors[(t_idx * 2 + c_idx) % 4], width=2)

    # ground-truth GR (or first curve) vs depth beside it
    las = lasio.read(lasf[0])
    depth = pd.to_numeric(np.asarray(las.index), errors="coerce")
    gr_name = next((c.mnemonic for c in las.curves
                    if "GR" in c.mnemonic.upper()), las.curves[1].mnemonic)
    gr = pd.to_numeric(np.asarray(las[gr_name]), errors="coerce")

    fig, ax = plt.subplots(1, 2, figsize=(11, 13),
                           gridspec_kw={"width_ratios": [2, 1]})
    ax[0].imshow(small)
    ax[0].set_title(f"{well}  scan + traced curves\n"
                    f"depth {cal.depth_at(layout.log_top):.0f}-"
                    f"{cal.depth_at(layout.log_bottom):.0f} ft, "
                    f"RMS {cal.rms_residual_ft:.2f} ft", fontsize=9)
    ax[0].axis("off")
    ax[1].plot(gr, depth, "g", lw=0.4)
    ax[1].set_title(f"ground truth: {gr_name}", fontsize=9)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(gr_name)
    ax[1].set_ylabel("depth ft")
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(f"{OUT}/{well}.png", dpi=70)
    plt.close()
    print(f"{well}: done", flush=True)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        try:
            overlay(w)
        except Exception as e:
            print(f"{w}: FAIL {str(e)[:60]}", flush=True)
