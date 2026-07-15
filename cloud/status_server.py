"""Live progress page for the cloud run. Stdlib only.

Usage (on the box):
    python cloud/status_server.py --port 8000
Then open http://<instance-ip>:<mapped-port>/
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = time.time()


def newest_log() -> str:
    logs = sorted(glob.glob(os.path.join(ROOT, "cycle*.log")) +
                  glob.glob(os.path.join(ROOT, "train_*.log")) +
                  glob.glob(os.path.join(ROOT, "benchmark*.log")),
                  key=lambda p: os.path.getmtime(p))
    return logs[-1] if logs else os.path.join(ROOT, "cycle1.log")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"(err: {e})"


def tail(path: str, n: int = 25) -> str:
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(not found)"


def manifest_stats() -> tuple[int, int]:
    path = os.path.join(ROOT, "data", "label_crops", "manifest.csv")
    try:
        with open(path) as f:
            rows = list(csv.reader(f))[1:]
        return len(rows), len({r[2] for r in rows if len(r) > 2})
    except OSError:
        return 0, 0


_last = {"t": 0.0, "crops": 0, "rate": 0.0}


def crop_rate(crops: int) -> float:
    now = time.time()
    if _last["t"] and now > _last["t"]:
        inst = (crops - _last["crops"]) / (now - _last["t"]) * 3600
        if crops > _last["crops"]:
            _last["rate"] = 0.5 * _last["rate"] + 0.5 * inst if _last["rate"] else inst
    _last["t"], _last["crops"] = now, crops
    return _last["rate"]


def stage() -> str:
    # read the WHOLE log for stage markers (they scroll out of any tail)
    try:
        log = open(newest_log(), errors="replace").read()
    except OSError:
        return "starting..."
    bench = os.path.join(ROOT, "data", "benchmark.csv")
    if "Traceback" in log[-4000:]:
        return "ERROR (see log tail)"
    if os.path.exists(bench):
        return "BENCHMARK running"
    if "== retrain" in log:
        return "TRAINING model"
    if "wells to harvest" in log:
        return "HARVEST running"
    return "starting..."


_prog = {"t": 0.0, "idx": 0, "rate": 0.0}


def progress() -> tuple[float, str]:
    """(percent done, ETA) from the last contributing well's position in
    the sorted well list — imap results arrive roughly in submission order,
    so alphabetical position tracks the sweep without touching the workers.
    """
    try:
        dirs = sorted(os.listdir(os.path.join(ROOT, "data", "pairs")))
        with open(os.path.join(ROOT, "data", "label_crops",
                               "manifest.csv")) as f:
            last_well = list(csv.reader(f))[-1][2]
        idx = dirs.index(last_well) + 1
        total = len(dirs)
    except (OSError, ValueError, IndexError):
        return 0.0, "-"

    now = time.time()
    if _prog["t"] and idx > _prog["idx"]:
        inst = (idx - _prog["idx"]) / (now - _prog["t"])  # wells/sec
        _prog["rate"] = (0.5 * _prog["rate"] + 0.5 * inst
                         if _prog["rate"] else inst)
    if idx > _prog["idx"] or not _prog["t"]:
        _prog["t"], _prog["idx"] = now, idx

    pct = 100.0 * idx / max(1, total)
    if _prog["rate"] > 0:
        secs = (total - idx) / _prog["rate"]
        eta = f"{int(secs//3600)}h {int(secs%3600//60)}m"
    else:
        eta = "measuring..."
    return pct, eta


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>LogLift cloud run</title>
<style>
 body{{font:15px/1.5 system-ui;margin:0;background:#0e1720;color:#dce6ee}}
 header{{background:#132330;padding:14px 26px;border-bottom:1px solid #1f3547}}
 h1{{margin:0;font-size:18px}} main{{max-width:900px;margin:20px auto;padding:0 16px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:18px}}
 .card{{background:#132330;border:1px solid #1f3547;border-radius:9px;padding:14px}}
 .card b{{display:block;font-size:23px;margin-top:3px;color:#7fd4a8}}
 .stage{{font-size:17px;padding:10px 16px;border-radius:8px;background:#1a3a2a;
        border:1px solid #2c5c42;margin-bottom:16px;display:inline-block}}
 pre{{background:#0a121a;border:1px solid #1f3547;border-radius:8px;padding:12px;
     overflow-x:auto;font-size:12.5px;white-space:pre-wrap}}
 .muted{{color:#8aa3b8;font-size:13px}}
</style></head><body>
<header><h1>LogLift — cloud run live status</h1>
<span class="muted">auto-refreshes every 15 s · {now}</span></header>
<main>
<div class="stage">STAGE: {stage}</div>
<div class="grid">
 <div class="card">Harvest progress<b>{pct:.1f} %</b></div>
 <div class="card">Est. time left (this stage)<b>{eta}</b></div>
 <div class="card">Wells contributed<b>{wells:,}</b></div>
 <div class="card">Training crops<b>{crops:,}</b></div>
</div>
<div class="grid">
 <div class="card">Crops / hour<b>{rate:,.0f}</b></div>
 <div class="card">Server uptime<b>{up}</b></div>
</div>
<div class="grid">
 <div class="card">GPU 0<b>{gpu0}</b></div>
 <div class="card">GPU 1<b>{gpu1}</b></div>
 <div class="card">CPU load<b>{load}</b></div>
 <div class="card">Disk used<b>{disk}</b></div>
</div>
<h3>Latest log</h3><pre>{log}</pre>
<h3>Training log (when active)</h3><pre>{train}</pre>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        crops, wells = manifest_stats()
        pct, eta = progress()
        gpus = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used"
                  " --format=csv,noheader").splitlines() or ["?", "?"]
        up = int(time.time() - START)
        body = PAGE.format(
            now=time.strftime("%H:%M:%S UTC", time.gmtime()),
            stage=html.escape(stage()),
            pct=pct, eta=html.escape(eta),
            wells=wells, crops=crops, rate=crop_rate(crops),
            up=f"{up//3600}h {up%3600//60}m",
            gpu0=html.escape(gpus[0] if gpus else "?"),
            gpu1=html.escape(gpus[1] if len(gpus) > 1 else "?"),
            load=sh("cut -d' ' -f1 /proc/loadavg"),
            disk=sh(f"du -sh {ROOT}/data 2>/dev/null | cut -f1"),
            log=html.escape(tail(newest_log())),
            train=html.escape(tail(os.path.join(ROOT, "data",
                                                "overnight_train.log"), 12)
                              if os.path.exists(os.path.join(
                                  ROOT, "data", "overnight_train.log"))
                              else sh(f"grep -E 'synth|real ' {ROOT}/cycle1.log"
                                      " | tail -12")),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
