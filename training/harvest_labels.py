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
    python -m training.harvest_labels
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


def harvest(pairs_dir: str = "data/pairs") -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    done_wells = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            done_wells = {r["well"] for r in csv.DictReader(f)}

    new_file = not os.path.exists(MANIFEST)
    n_crops = n_wells = 0
    with open(MANIFEST, "a", newline="") as mf:
        writer = csv.writer(mf)
        if new_file:
            writer.writerow(["path", "value", "well", "dpi", "rms_ft"])

        for well_dir in sorted(glob.glob(os.path.join(pairs_dir, "*"))):
            well = os.path.basename(well_dir)
            if well in done_wells:
                continue
            scans = glob.glob(os.path.join(well_dir, "scan_*"))
            if not scans:
                continue
            scan = scans[0]
            try:
                layout = detect_layout(scan)
                cal = calibrate_depth(scan, layout)
            except Exception:
                continue
            if cal.depth_band is None or cal.rms_residual_ft > 2.0:
                continue

            gray = load_gray(scan)
            dpi = layout.width / 8.25
            half = int(0.22 * dpi)          # label crop half-height
            left, right = cal.depth_band

            for i, (row, value) in enumerate(cal.labels):
                r0 = max(0, int(row) - half)
                r1 = min(gray.shape[0], int(row) + half)
                crop = gray[r0:r1, left:right]
                if crop.size == 0:
                    continue
                name = f"{int(value)}_{well}_{i}.png"
                path = os.path.join(OUT_DIR, name)
                Image.fromarray(crop).save(path)
                writer.writerow([path, int(value), well,
                                 round(dpi), round(cal.rms_residual_ft, 2)])
                n_crops += 1
            n_wells += 1
            mf.flush()
            print(f"{well}: {len(cal.labels)} crops", flush=True)

    print(f"harvested {n_crops} crops from {n_wells} new wells -> {OUT_DIR}")


if __name__ == "__main__":
    harvest()
