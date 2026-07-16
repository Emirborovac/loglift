"""Playground server (port 3000). For now: just show the whole log.

We iterate from here under your guidance, step by step.
    python -m playground.server
"""

from __future__ import annotations

import glob
import io
import os

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, JSONResponse
from PIL import Image

from scipy.ndimage import binary_opening, median_filter

from extraction.layout import detect_layout, load_gray

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(__file__)
app = FastAPI()

# the one log we are working on
WELL = os.path.join(HERE, "..", "data", "pairs", "15001273970000")
SCAN = glob.glob(os.path.join(WELL, "scan_*"))[0]


def leftmost_curve(thr: int = 115, vthick: int = 4, tol: float = 9.0):
    """Clean, approved version: leftmost edge of the first curve in track 1.

    Vertical opening removes thin horizontal grid lines (curve is thicker);
    a left-outlier reject removes residual bold grid lines. This is CLEAN
    but slightly under-reads very sharp rightward peaks (a known TODO we
    tackle separately).
    """
    lay = detect_layout(SCAN)
    gray = load_gray(SCAN)
    l, r = lay.tracks[0]
    top, bot = lay.log_top, lay.log_bottom
    band = gray[top:bot, l:r]
    H, W = band.shape

    barrier = band < thr
    clean = binary_opening(barrier, structure=np.ones((vthick, 1)))  # kill grid

    coldark = clean.mean(axis=0)
    skip = 0
    while skip < W // 4 and coldark[skip] > 0.5:
        skip += 1
    skip = max(skip + 2, 4)

    idx = np.arange(H)
    xs = np.full(H, np.nan)
    for i in range(H):
        w = np.where(clean[i, skip:])[0]
        if len(w):
            xs[i] = w[0] + skip
    g = ~np.isnan(xs)
    if g.sum() < 2:
        return {"points": [], "left_x": float(l + skip)}
    xs = np.interp(idx, idx[g], xs[g])

    ref = median_filter(xs, size=41)
    xs[xs < ref - tol] = np.nan
    g = ~np.isnan(xs)
    xs = np.interp(idx, idx[g], xs[g])

    step = max(1, H // 4000)
    pts = [[float(l + xs[i]), float(top + i)] for i in range(0, H, step)]
    return {"points": pts, "left_x": float(l + skip)}


@app.get("/curve1.json")
def curve1(thr: int = 115, vthick: int = 4, tol: float = 9.0):
    return JSONResponse(leftmost_curve(thr, vthick, tol))

@app.get("/log.png")
def log_png():
    # full native resolution so zooming reveals real detail
    im = Image.open(SCAN).convert("L")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/meta")
def meta():
    im = Image.open(SCAN)
    return {"w": im.width, "h": im.height,
            "file": os.path.basename(SCAN)}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="warning")
