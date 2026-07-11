"""Match each scan image to the LAS file covering the same depth interval.

A well often has several scans (different tools/runs) and several LAS files.
A scan is only a useful training pair if a LAS file actually covers the depth
interval shown on the image. We match on depth overlap:

    coverage = overlap(scan interval, las interval) / scan interval length

and keep the best-covering LAS per scan, flagging pairs below a threshold.

Input:  data/manifest_pairs.csv   (from pipeline.indexes)
Output: data/matched_pairs.csv    one row per scan with its best LAS match

Usage:
    python -m pipeline.pairs [--min-coverage 0.8]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "manifest_pairs.csv"
OUT = DATA_DIR / "matched_pairs.csv"


def _to_float(value) -> float | None:
    try:
        v = float(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def best_las_match(scan_top: float, scan_bot: float, las_files: str):
    """Return (url, coverage, las_top, las_bot) of the LAS best covering the scan."""
    best = (None, 0.0, None, None)
    for entry in str(las_files).split(";"):
        parts = entry.split("|")
        if len(parts) != 3 or not parts[0]:
            continue
        url, start, stop = parts
        top, bot = _to_float(start), _to_float(stop)
        if top is None or bot is None:
            continue
        top, bot = min(top, bot), max(top, bot)
        overlap = min(scan_bot, bot) - max(scan_top, top)
        coverage = max(0.0, overlap) / (scan_bot - scan_top)
        if coverage > best[1]:
            best = (url, coverage, top, bot)
    return best


def build_matched_pairs(min_coverage: float = 0.8) -> pd.DataFrame:
    df = pd.read_csv(MANIFEST, dtype=str)

    rows = []
    for _, r in df.iterrows():
        scan_top = _to_float(r["LOG_TOP"])
        scan_bot = _to_float(r["LOG_BOTTOM"])
        if scan_top is None or scan_bot is None or scan_bot - scan_top < 100:
            continue  # no usable depth interval on the scan record
        scan_top, scan_bot = min(scan_top, scan_bot), max(scan_top, scan_bot)

        url, coverage, las_top, las_bot = best_las_match(scan_top, scan_bot, r["LAS_FILES"])
        if url is None:
            continue
        rows.append({
            "API_NUM_NODASH": r["API_NUM_NODASH"],
            "LEASE_AND_WELL": r["LEASE_AND_WELL"],
            "TOOL": r["TOOL"],
            "LOGGER": r["LOGGER"],
            "LOG_DATE": r["LOG_DATE"],
            "SCAN_TOP": scan_top,
            "SCAN_BOTTOM": scan_bot,
            "SCAN_URL": r["SCAN_URL"],
            "SCAN_URL_ORIG": r.get("SCAN_URL_ORIG", ""),
            "LAS_URL": url,
            "LAS_TOP": las_top,
            "LAS_BOTTOM": las_bot,
            "COVERAGE": round(coverage, 3),
            "GOOD_PAIR": coverage >= min_coverage,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    good = out[out["GOOD_PAIR"]]
    print(f"scans with a depth interval:        {len(out):,}")
    print(f"good pairs (coverage >= {min_coverage:.0%}):     {len(good):,} "
          f"across {good['API_NUM_NODASH'].nunique():,} wells")
    print(f"wrote -> {OUT}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="minimum fraction of the scan interval a LAS must cover")
    args = parser.parse_args()
    build_matched_pairs(min_coverage=args.min_coverage)
