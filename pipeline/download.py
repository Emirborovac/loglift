"""Download matched scan (TIFF) + LAS pairs from data/matched_pairs.csv.

Each downloaded pair lands in data/pairs/<API_NUM_NODASH>/:

    <API>/
        scan_<id>.tif      the scanned paper log image
        las_<id>.las       the LAS file covering the same depth interval
                           (ground truth)

Run pipeline.indexes then pipeline.pairs first.

Usage:
    python -m pipeline.download --limit 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "manifest_pairs.csv"
MATCHED = DATA_DIR / "matched_pairs.csv"
PAIRS_DIR = DATA_DIR / "pairs"

TIMEOUT = 60


def _swap_ext_case(url: str) -> str:
    base, dot, ext = url.rpartition(".")
    if not dot:
        return url
    return f"{base}.{ext.lower() if ext.isupper() else ext.upper()}"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code != 200 or "html" in resp.headers.get("Content-Type", "").lower():
            return False
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def download_pairs(limit: int = 20, pairs_path: Path = MATCHED) -> None:
    """Download matched (scan, LAS) pairs produced by pipeline.pairs."""
    df = pd.read_csv(pairs_path)
    df = df[df["GOOD_PAIR"]]

    # one scan row per well first, so a small --limit spans many wells
    wells = df.drop_duplicates("API_NUM_NODASH").head(limit)

    ok_scans = ok_las = 0
    for _, row in tqdm(wells.iterrows(), total=len(wells), desc="pairs"):
        well_dir = PAIRS_DIR / str(row["API_NUM_NODASH"])
        well_dir.mkdir(parents=True, exist_ok=True)

        scan_name = row["SCAN_URL"].rstrip("/").rsplit("/", 1)[-1]
        dest = well_dir / f"scan_{scan_name}"
        # Azure blob names are case-sensitive and extension case varies (.tif/.TIF),
        # so try the rewritten URL, the original KGS URL, and both extension cases.
        candidates = []
        for url in (row["SCAN_URL"], row.get("SCAN_URL_ORIG")):
            if isinstance(url, str) and url:
                candidates += [url, _swap_ext_case(url)]
        if any(_download(url, dest) for url in candidates):
            ok_scans += 1

        las_url = row["LAS_URL"]
        las_name = las_url.rstrip("/").rsplit("/", 1)[-1]
        if _download(las_url, well_dir / f"las_{las_name}"):
            ok_las += 1

    print(f"downloaded: {ok_scans}/{len(wells)} scans, {ok_las}/{len(wells)} LAS files -> {PAIRS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="number of wells to download")
    args = parser.parse_args()
    download_pairs(limit=args.limit)
