"""LogLift web app: upload a scanned well log, get a LAS file back.

Run:
    uvicorn app.main:app --port 8517
"""

from __future__ import annotations

import io
import os
import threading
import traceback
import uuid
import warnings

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from PIL import Image, ImageDraw

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves
from extraction.header import read_header
from convert import _assign_curves
from export.las_writer import write_las

warnings.filterwarnings("ignore")

app = FastAPI(title="LogLift")

JOBS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

JOBS: dict[str, dict] = {}  # job_id -> {status, message, ...}


def _overlay_png(gray, layout, track_curves, out_path, width=900):
    """Traced curves drawn over the (downsampled) scan for visual review."""
    im = Image.fromarray(gray).convert("RGB")
    scale = width / im.width
    im = im.resize((width, int(im.height * scale)))
    d = ImageDraw.Draw(im)
    colors = [(230, 40, 40), (30, 110, 230), (30, 160, 60), (200, 120, 20)]

    for t_idx, curves in track_curves.items():
        left, right = layout.tracks[t_idx]
        span = (right - left - 12) * scale
        x0 = (left + 6) * scale
        for c_idx, spec in enumerate(curves):
            vals = spec["values"] if isinstance(spec, dict) else spec
            norm = spec.get("norm") if isinstance(spec, dict) else None
            arr = np.asarray(norm if norm is not None else vals, dtype=float)
            if arr.max() > 1.5:  # scaled curve: re-normalize for drawing
                lo, hi = np.nanmin(arr), np.nanmax(arr)
                arr = (arr - lo) / max(1e-9, hi - lo)
            color = colors[(t_idx * 2 + c_idx) % len(colors)]
            step = max(1, int(2 / scale))
            pts = []
            for row in range(0, len(arr), step):
                y = (layout.log_top + row) * scale
                pts.append((x0 + float(arr[row]) * span, y))
            if len(pts) > 1:
                d.line(pts, fill=color, width=2)
    im.save(out_path)


def _run_job(job_id: str, scan_path: str, well: str, api: str):
    job = JOBS[job_id]
    try:
        job.update(status="running", message="detecting layout...")
        layout = detect_layout(scan_path)
        if not layout.tracks and not layout.depth_col_candidates:
            raise ValueError("could not detect a log layout on this image")

        job["message"] = "calibrating depth (OCR)..."
        cal = calibrate_depth(scan_path, layout)
        layout = reanchor_tracks(layout, cal.depth_band)

        job["message"] = "reading header..."
        try:
            scales = read_header(scan_path, layout)
        except Exception:
            scales = []

        job["message"] = "tracing curves..."
        gray = load_gray(scan_path)
        track_curves = {}
        for t_idx, track in enumerate(layout.tracks):
            traces = extract_track_curves(
                gray, track, layout.log_top, layout.log_bottom, n_curves=2)
            t_scales = [s for s in scales if s.track == t_idx]
            specs = _assign_curves(traces, t_scales)
            # keep the normalized trace for overlay drawing
            for spec, tr in zip(specs, traces):
                if isinstance(spec, dict):
                    spec["norm"] = tr
            track_curves[t_idx] = specs

        job["message"] = "writing LAS..."
        las_path = os.path.join(JOBS_DIR, f"{job_id}.las")
        write_las(las_path, layout, cal,
                  {t: [({k: v for k, v in s.items() if k != "norm"}
                        if isinstance(s, dict) else s) for s in specs_]
                   for t, specs_ in track_curves.items()},
                  well_name=well, api=api,
                  source_scan=os.path.basename(scan_path))

        overlay_path = os.path.join(JOBS_DIR, f"{job_id}.png")
        _overlay_png(gray, layout, track_curves, overlay_path)

        d0 = cal.depth_at(layout.log_top)
        d1 = cal.depth_at(layout.log_bottom)
        curves_out = []
        for t_idx, specs_ in track_curves.items():
            for s in specs_:
                if isinstance(s, dict):
                    curves_out.append(dict(
                        name=s["name"], unit=s["unit"],
                        scaled=s["unit"] != "norm",
                        verify="?" in s["name"]))
        job.update(
            status="done", message="complete",
            report=dict(
                depth_top_ft=round(d0, 1), depth_bottom_ft=round(d1, 1),
                depth_labels_used=cal.n_inliers,
                depth_rms_ft=round(cal.rms_residual_ft, 2),
                tracks=len(layout.tracks),
                curves=curves_out,
            ))
    except Exception as e:
        traceback.print_exc()
        job.update(status="failed",
                   message=f"{type(e).__name__}: {e}")


@app.post("/convert")
async def convert_endpoint(file: UploadFile = File(...),
                           well: str = Form(""), api: str = Form("")):
    job_id = uuid.uuid4().hex[:12]
    scan_path = os.path.join(JOBS_DIR, f"{job_id}_{file.filename}")
    with open(scan_path, "wb") as f:
        f.write(await file.read())
    JOBS[job_id] = dict(status="queued", message="queued")
    threading.Thread(target=_run_job,
                     args=(job_id, scan_path, well, api), daemon=True).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return JSONResponse(job)


@app.get("/result/{job_id}/las")
def result_las(job_id: str):
    path = os.path.join(JOBS_DIR, f"{job_id}.las")
    if not os.path.exists(path):
        raise HTTPException(404, "no LAS for this job")
    return FileResponse(path, filename=f"loglift_{job_id}.las",
                        media_type="text/plain")


@app.get("/result/{job_id}/overlay")
def result_overlay(job_id: str):
    path = os.path.join(JOBS_DIR, f"{job_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "no overlay for this job")
    return FileResponse(path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"),
              encoding="utf-8") as f:
        return f.read()
