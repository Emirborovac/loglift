"""Harvest depth-label crops as training data for a custom digit reader.

For every well whose depth calibration succeeds, each inlier label gives a
(row, value) pair we trust: the value survived round-number filtering,
slope consensus AND the physical-scale gate. Crop the label's image patch
and store it with that value — a self-labeling loop: today's successes
train tomorrow's model, which unlocks harder wells, which yield more
training data.

Output:
    data/label_crops/<value>_<well>_<n>.png
    data/label_crops/manifest.csv   (path, value, well, dpi, rms_ft)

Usage:
    python -m training.harvest_labels [--workers N]

Workers are separate processes, each with its own OCR model (~1 GB VRAM).
"""

from __future__ import annotations

import csv
import glob
import os
import warnings

from PIL import Image

from extraction.layout import detect_layout, load_gray
from extraction.depth import calibrate_depth

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "label_crops")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")


def harvest_well(well_dir: str) -> list[tuple[str, int, str, int, float]]:
    """Process one well; saves crop PNGs, returns manifest rows."""
    well = os.path.basename(well_dir)
    scans = glob.glob(os.path.join(well_dir, "scan_*"))
    if not scans:
        return []
    scan = scans[0]
    try:
        layout = detect_layout(scan)
        cal = calibrate_depth(scan, layout)
    except Exception:
        return []
    if cal.depth_band is None or cal.rms_residual_ft > 2.0:
        return []

    gray = load_gray(scan)
    dpi = layout.width / 8.25
    half = int(0.22 * dpi)
    left, right = cal.depth_band

    rows = []
    for i, (row, value) in enumerate(cal.labels):
        r0 = max(0, int(row) - half)
        r1 = min(gray.shape[0], int(row) + half)
        crop = gray[r0:r1, left:right]
        if crop.size == 0:
            continue
        path = os.path.join(OUT_DIR, f"{int(value)}_{well}_{i}.png")
        Image.fromarray(crop).save(path)
        rows.append((path, int(value), well, round(dpi),
                     round(cal.rms_residual_ft, 2)))
    return rows


def harvest(pairs_dir: str = "data/pairs", workers: int = 1) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    done_wells = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            done_wells = {r["well"] for r in csv.DictReader(f)}

    todo = [d for d in sorted(glob.glob(os.path.join(pairs_dir, "*")))
            if os.path.basename(d) not in done_wells]
    print(f"{len(todo)} wells to harvest, {workers} workers", flush=True)

    new_file = not os.path.exists(MANIFEST)
    n_crops = n_wells = 0
    with open(MANIFEST, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(["path", "value", "well", "dpi", "rms_ft"])

        def consume(rows):
            nonlocal n_crops, n_wells
            if not rows:
                return
            for r in rows:
                writer.writerow(r)
            mf.flush()
            n_crops += len(rows)
            n_wells += 1
            print(f"{rows[0][2]}: {len(rows)} crops", flush=True)

        if workers <= 1:
            for well_dir in todo:
                consume(harvest_well(well_dir))
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # own OCR model per worker
            with ctx.Pool(workers) as pool:
                for rows in pool.imap_unordered(harvest_well, todo):
                    consume(rows)

    print(f"harvested {n_crops} crops from {n_wells} new wells -> {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    harvest(workers=a.workers)
