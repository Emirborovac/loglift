"""Playground server (port 3000): a single-log sandbox for iterating on the
curve-extraction algorithm and seeing the result live on a real track.

Requires at least one downloaded well (see the repo README quick-start):
    python -m pipeline.download --limit 50
Then:
    python -m playground.server        # open http://localhost:3000
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

# the log to work on: a specific well if present, else the first available.
# Lazy so the module imports even before any data is downloaded.
_PAIRS = os.path.join(HERE, "..", "data", "pairs")


def _scan_path() -> str | None:
    prefer = os.path.join(_PAIRS, "15001273970000")
    hits = glob.glob(os.path.join(prefer, "scan_*"))
    if not hits:
        hits = glob.glob(os.path.join(_PAIRS, "*", "scan_*"))
    return hits[0] if hits else None


def leftmost_curve(thr: int = 115, vthick: int = 4, tol: float = 9.0):
    """Clean, approved version: leftmost edge of the first curve in track 1.

    Vertical opening removes thin horizontal grid lines (curve is thicker);
    a left-outlier reject removes residual bold grid lines. This is CLEAN
    but slightly under-reads very sharp rightward peaks (a known TODO we
    tackle separately).
    """
    scan = _scan_path()
    if scan is None:
        return {"points": [], "left_x": 0.0}
    lay = detect_layout(scan)
    gray = load_gray(scan)
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
    scan = _scan_path()
    if scan is None:
        return Response(b"", media_type="image/png")
    im = Image.open(scan).convert("L")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/meta")
def meta():
    scan = _scan_path()
    if scan is None:
        return {"w": 0, "h": 0, "file": "no data - run pipeline.download first"}
    im = Image.open(scan)
    return {"w": im.width, "h": im.height,
            "file": os.path.basename(scan)}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="warning")
