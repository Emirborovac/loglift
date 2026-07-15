"""End-to-end benchmark: scan -> layout -> depth -> curves vs LAS truth.

For every downloaded pair:
1. detect layout, calibrate depth
2. trace 2 curves per track
3. correlate each trace against every LAS curve (small depth-lag search)
4. record the best |r| match per trace

Output: data/benchmark.csv with one row per traced curve.

Usage:
    python benchmark.py [--workers 3]

Workers are separate processes, each with its own OCR model (~1 GB VRAM);
3 workers fit an 8 GB GPU comfortably.
"""

from __future__ import annotations

import glob
import os
import warnings

import lasio
import numpy as np
import pandas as pd

from extraction.layout import detect_layout, load_gray, reanchor_tracks
from extraction.depth import calibrate_depth
from extraction.curves import extract_track_curves

warnings.filterwarnings("ignore")


def best_las_match(trace: np.ndarray, scan_depths: np.ndarray,
                   las) -> tuple[str, float, float]:
    """(mnemonic, r, lag_ft) of the LAS curve best matching the trace."""
    depths = pd.to_numeric(np.asarray(las.index), errors="coerce")
    best = ("", 0.0, 0.0)
    for curve in las.curves[1:]:
        # some LAS files carry non-standard off-scale markers ('********',
        # '#####') instead of the declared NULL; coerce them to NaN
        vals = pd.to_numeric(np.asarray(las[curve.mnemonic]),
                             errors="coerce")
        ok = ~np.isnan(vals)
        if ok.sum() < 100:
            continue
        for lag in np.arange(-10, 10.5, 2.5):
            gt = np.interp(scan_depths + lag, depths[ok], vals[ok],
                           left=np.nan, right=np.nan)
            v = ~np.isnan(gt)
            if v.sum() < 500:
                continue
            r = float(np.corrcoef(trace[v], gt[v])[0, 1])
            if abs(r) > abs(best[1]):
                best = (curve.mnemonic, r, float(lag))
    return best


COLUMNS = ["well", "stage", "error", "track", "curve", "trace_std",
           "match", "r", "lag_ft", "rms_ft"]


def bench_well(well_dir: str) -> list[dict]:
    """Wrapper: no single well may ever crash the pool."""
    try:
        return _bench_well(well_dir)
    except Exception as e:
        return [dict(well=os.path.basename(well_dir), stage="crash",
                     error=f"{type(e).__name__}: {str(e)[:60]}")]


def _bench_well(well_dir: str) -> list[dict]:
    """Run the full pipeline on one well; returns result/error rows."""
    well = os.path.basename(well_dir)
    scans = glob.glob(os.path.join(well_dir, "scan_*"))
    las_files = glob.glob(os.path.join(well_dir, "las_*"))
    if not scans or not las_files:
        return []
    scan = scans[0]
    rows: list[dict] = []

    try:
        layout = detect_layout(scan)
        cal = calibrate_depth(scan, layout)
        layout = reanchor_tracks(layout, cal.depth_band)
    except Exception as e:
        return [dict(well=well, stage="depth",
                     error=f"{type(e).__name__}: {str(e)[:60]}")]

    try:
        las = lasio.read(las_files[0])
    except Exception as e:
        return [dict(well=well, stage="las",
                     error=f"{type(e).__name__}: {str(e)[:60]}")]

    gray = load_gray(scan)
    scan_rows = np.arange(layout.log_top, layout.log_bottom)
    scan_depths = cal.slope * scan_rows + cal.intercept

    for t_idx, track in enumerate(layout.tracks):
        try:
            traces = extract_track_curves(
                gray, track, layout.log_top, layout.log_bottom, n_curves=2)
        except Exception as e:
            rows.append(dict(well=well, stage=f"trace_t{t_idx}",
                             error=f"{type(e).__name__}: {str(e)[:60]}"))
            continue
        for c_idx, trace in enumerate(traces):
            mnem, r, lag = best_las_match(trace, scan_depths, las)
            rows.append(dict(
                well=well, stage="ok", track=t_idx, curve=c_idx,
                trace_std=round(float(np.std(trace)), 3),
                match=mnem, r=round(r, 3), lag_ft=lag,
                rms_ft=round(cal.rms_residual_ft, 2),
            ))
    return rows


def run(pairs_dir: str = "data/pairs", out_csv: str = "data/benchmark.csv",
        workers: int = 3, limit: int | None = None):
    # resume: skip wells already in the output CSV; results append per well,
    # so an interrupted run keeps its progress
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv, dtype=str).well)

    todo = []
    for well_dir in sorted(glob.glob(os.path.join(pairs_dir, "*"))):
        if os.path.basename(well_dir) in done:
            continue
        if (glob.glob(os.path.join(well_dir, "scan_*"))
                and glob.glob(os.path.join(well_dir, "las_*"))):
            todo.append(well_dir)
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} wells to benchmark, {workers} workers", flush=True)

    def flush(rows):
        if rows:
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(
                out_csv, mode="a", index=False,
                header=not os.path.exists(out_csv))

    if workers <= 1:
        for wd in todo:
            flush(bench_well(wd))
            print(f"{os.path.basename(wd)}: done", flush=True)
    else:
        import multiprocessing as mp
        from training.harvest_labels import worker_init
        n_gpus = int(os.environ.get("N_GPUS", "1"))
        ctx = mp.get_context("spawn")  # each worker gets its own OCR model
        counter = ctx.Value("i", 0)
        with ctx.Pool(workers, initializer=worker_init,
                      initargs=(counter, n_gpus)) as pool:
            for rows in pool.imap_unordered(bench_well, todo):
                flush(rows)
                if rows:
                    print(f"{rows[0]['well']}: done", flush=True)

    df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    ok = df[df.stage == "ok"]
    if len(ok):
        strong = ok[ok.r.abs() >= 0.7]
        print(f"\ntraces benchmarked: {len(ok)} across {ok.well.nunique()} wells")
        print(f"strong matches (|r|>=0.7): {len(strong)} "
              f"({len(strong)/len(ok):.0%}) on {strong.well.nunique()} wells")
        print("match mnemonics:", strong.match.value_counts().to_dict())
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    run(workers=a.workers, limit=a.limit)
