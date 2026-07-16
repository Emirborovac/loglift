"""LogLift assisted digitizer — upload a scan, auto-extract, correct by hand.

The honest, industry-standard workflow: the app auto-traces to give a head
start; the user drags the curves to match the scan exactly, sets curve
names/scales, and exports a clean LAS.

Run:
    uvicorn app.main:app --port 8517
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
import warnings

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves
from extraction.header import read_header

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title="LogLift")
HERE = os.path.dirname(__file__)
JOBS_DIR = os.path.join(HERE, "..", "data", "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
JOBS: dict[str, dict] = {}

DISPLAY_W = 620            # scan display width in px
N_CTRL = 60               # control points per curve
CURVE_COLORS = ["#e63030", "#1f7be6", "#1ea832", "#d67800",
                "#8e44ad", "#00998e"]


def _decimate(norm: np.ndarray, n: int) -> list[int]:
    """Indices of n evenly spaced control points."""
    return list(np.linspace(0, len(norm) - 1, n).astype(int))


def _run_job(job_id: str, scan_path: str, well: str, api: str):
    job = JOBS[job_id]
    try:
        job.update(status="running", message="detecting layout…")
        layout = detect_layout(scan_path)
        if not layout.tracks and not layout.depth_col_candidates:
            raise ValueError("could not detect a log layout on this image")

        job["message"] = "calibrating depth (OCR)…"
        cal = calibrate_depth(scan_path, layout)
        layout = reanchor_tracks(layout, cal.depth_band)

        job["message"] = "reading header…"
        try:
            scales = read_header(scan_path, layout)
        except Exception:
            scales = []

        job["message"] = "tracing curves…"
        gray = load_gray(scan_path)
        H0, W0 = gray.shape
        scale = DISPLAY_W / W0
        disp_h = int(H0 * scale)

        # downsampled scan for the editor background
        disp = Image.fromarray(gray).resize((DISPLAY_W, disp_h))
        disp.convert("L").save(os.path.join(JOBS_DIR, f"{job_id}_scan.png"))

        curves = []
        ci = 0
        for t_idx, track in enumerate(layout.tracks):
            traces = extract_track_curves(gray, track, layout.log_top,
                                          layout.log_bottom, n_curves=2)
            tscales = [s for s in scales if s.track == t_idx]
            for k, tr in enumerate(traces):
                left, right = track
                rows = np.arange(layout.log_top, layout.log_bottom)
                idx = _decimate(tr, N_CTRL)
                pts = [[round((left + 6 + tr[i] * (right - left - 12)) * scale, 1),
                        round((layout.log_top + i) * scale, 1)] for i in idx]
                sc = tscales[k] if k < len(tscales) else None
                curves.append(dict(
                    id=ci, name=(sc.mnemonic if sc else f"CURVE{ci+1}"),
                    unit=(sc.unit if sc else ""),
                    left_value=(sc.left_value if sc and sc.left_value is not None else 0.0),
                    right_value=(sc.right_value if sc and sc.right_value is not None else 100.0),
                    color=CURVE_COLORS[ci % len(CURVE_COLORS)],
                    track=t_idx, points=pts))
                ci += 1

        data = dict(
            display_w=DISPLAY_W, display_h=disp_h, scale=scale,
            depth=dict(slope=cal.slope, intercept=cal.intercept,
                       top=cal.depth_at(layout.log_top),
                       bottom=cal.depth_at(layout.log_bottom),
                       rms=round(cal.rms_residual_ft, 2)),
            tracks=[[round(l * scale, 1), round(r * scale, 1)]
                    for l, r in layout.tracks],
            log_top=layout.log_top, log_bottom=layout.log_bottom,
            well=well, api=api, curves=curves)
        with open(os.path.join(JOBS_DIR, f"{job_id}_data.json"), "w") as f:
            json.dump(data, f)
        job.update(status="done", message="ready to review")
    except Exception as e:
        traceback.print_exc()
        job.update(status="failed", message=f"{type(e).__name__}: {e}")


def _start(scan_path: str, well: str, api: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = dict(status="queued", message="queued")
    threading.Thread(target=_run_job, args=(job_id, scan_path, well, api),
                     daemon=True).start()
    return job_id


@app.post("/convert")
async def convert(file: UploadFile = File(...), well: str = Form(""),
                  api: str = Form("")):
    job_id = uuid.uuid4().hex[:12]
    scan_path = os.path.join(JOBS_DIR, f"{job_id}_{file.filename}")
    with open(scan_path, "wb") as f:
        f.write(await file.read())
    JOBS[job_id] = dict(status="queued", message="queued")
    threading.Thread(target=_run_job, args=(job_id, scan_path, well, api),
                     daemon=True).start()
    return {"job_id": job_id}


import glob as _glob

SAMPLE_WELL = os.path.join(HERE, "..", "data", "pairs", "15001273970000")


@app.post("/sample")
def sample():
    """Kick off a job on a bundled example scan (Kansas, calibrates cleanly)."""
    scans = _glob.glob(os.path.join(SAMPLE_WELL, "scan_*"))
    if not scans:
        raise HTTPException(404, "sample scan not available in this install")
    return {"job_id": _start(scans[0], "CAMPBELL 20 (sample)", "15-001-27397")}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return JSONResponse(job)


@app.get("/result/{job_id}/scan.png")
def scan_png(job_id: str):
    p = os.path.join(JOBS_DIR, f"{job_id}_scan.png")
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/result/{job_id}/data.json")
def data_json(job_id: str):
    p = os.path.join(JOBS_DIR, f"{job_id}_data.json")
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json")


@app.post("/export/{job_id}")
def export(job_id: str, payload: dict = Body(...)):
    """Receive corrected curves (display coords) and write a LAS file."""
    import lasio
    p = os.path.join(JOBS_DIR, f"{job_id}_data.json")
    if not os.path.exists(p):
        raise HTTPException(404)
    meta = json.load(open(p))
    scale = meta["scale"]
    slope, intercept = meta["depth"]["slope"], meta["depth"]["intercept"]
    tracks = meta["tracks"]
    step = float(payload.get("step_ft", 0.5))

    curves = payload["curves"]
    # depth grid from the log extent
    d0, d1 = sorted([meta["depth"]["top"], meta["depth"]["bottom"]])
    d0 = np.ceil(d0 / step) * step
    d1 = np.floor(d1 / step) * step
    grid = np.arange(d0, d1 + step / 2, step)

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value=meta.get("well", ""))
    las.well["API"] = lasio.HeaderItem("API", value=meta.get("api", ""))
    las.well["STRT"] = lasio.HeaderItem("STRT", unit="FT", value=float(grid[0]))
    las.well["STOP"] = lasio.HeaderItem("STOP", unit="FT", value=float(grid[-1]))
    las.well["STEP"] = lasio.HeaderItem("STEP", unit="FT", value=step)
    las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)
    las.other = "Digitized with LogLift (assisted, human-corrected)."
    las.append_curve("DEPT", grid, unit="FT")

    for c in curves:
        pts = sorted(c["points"], key=lambda p: p[1])  # by display y
        if len(pts) < 2:
            continue
        ys = np.array([p[1] for p in pts]) / scale         # orig rows
        xs = np.array([p[0] for p in pts]) / scale         # orig x px
        depth = slope * ys + intercept
        tl, tr = tracks[c["track"]]
        tl, tr = tl / scale, tr / scale
        norm = (xs - (tl + 6)) / max(1.0, (tr - tl - 12))   # 0..1 in track
        lv, rv = float(c["left_value"]), float(c["right_value"])
        val = lv + norm * (rv - lv)
        order = np.argsort(depth)  # np.interp needs ascending x
        resampled = np.interp(grid, depth[order], val[order],
                              left=np.nan, right=np.nan)
        name = "".join(ch for ch in c["name"].upper()
                       if ch.isalnum() or ch in "_")[:8] or f"C{c['id']}"
        las.append_curve(name, resampled, unit=c.get("unit", ""))

    out = os.path.join(JOBS_DIR, f"{job_id}_corrected.las")
    with open(out, "w") as f:
        las.write(f, version=2.0)
    return FileResponse(out, filename=f"loglift_{job_id}.las",
                        media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()
