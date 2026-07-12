"""Measure the trained digit reader's lift on previously-failed wells.

Reads the v0 benchmark CSV, retries depth calibration (now model-assisted)
on every well that failed at the depth stage, and reports the recovery
rate.

Usage:
    python -m training.measure_lift
"""

from __future__ import annotations

import csv
import glob
import warnings

from extraction.layout import detect_layout
from extraction.depth import calibrate_depth
from extraction import digit_reader

warnings.filterwarnings("ignore")


def main():
    print("model available:", digit_reader.available(), flush=True)

    failed = set()
    with open("data/benchmark.csv") as f:
        for row in csv.reader(f):
            if row and row[0] != "well" and row[1] == "depth":
                failed.add(row[0])
    print(len(failed), "previously failed wells", flush=True)

    recovered = tried = 0
    for well in sorted(failed):
        scans = glob.glob(f"data/pairs/{well}/scan_*")
        if not scans:
            continue
        tried += 1
        try:
            lay = detect_layout(scans[0])
            cal = calibrate_depth(scans[0], lay)
            recovered += 1
            print(f"RECOVERED {well}: {cal.n_inliers} labels, "
                  f"rms {cal.rms_residual_ft:.2f} ft", flush=True)
        except Exception as e:
            print(f"still-fail {well}: {str(e)[:50]}", flush=True)

    print(f"RESULT: {recovered}/{tried} previously-failed wells now calibrate",
          flush=True)


if __name__ == "__main__":
    main()
