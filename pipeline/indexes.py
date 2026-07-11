"""Download the KGS scan/LAS indexes and build the paired-well manifest.

The Kansas Geological Survey publishes two bulk index files:

- ks_las_files.zip   -> one row per digital LAS file (with download URL)
- ks_elog_scans.zip  -> one row per scanned paper log image (with download URL,
                        tool type, and logged depth interval)

Both carry the well API number, so joining them yields the set of wells for
which we have BOTH a scan image and ground-truth digital curves.

KGS moved file storage to Azure blobs; the index URLs still point at the old
web server, so we rewrite them (verified working July 2026).

Usage:
    python -m pipeline.indexes            # writes data/manifest_pairs.csv
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

LAS_INDEX_URL = "https://www.kgs.ku.edu/PRS/Ora_Archive/ks_las_files.zip"
SCAN_INDEX_URL = "https://www.kgs.ku.edu/PRS/Ora_Archive/ks_elog_scans.zip"

# old web-server prefixes -> current Azure blob prefixes
URL_REWRITES = {
    "https://www.kgs.ku.edu/b_1/WebDocs/": "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/",
    "http://www.kgs.ku.edu/PRS/Scans/": "https://kgsimages.blob.core.windows.net/web/web_1/WebDocs/",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def rewrite_url(url: str) -> str:
    for old, new in URL_REWRITES.items():
        if url.startswith(old):
            return new + url[len(old):]
    return url


def _fetch_index(url: str) -> pd.DataFrame:
    """Download a KGS index zip and return its single CSV member as a DataFrame."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as f:
            return pd.read_csv(f, dtype=str, on_bad_lines="skip")


def build_manifest(out_path: Path | None = None) -> pd.DataFrame:
    """Join the two indexes on API number and write the paired-well manifest.

    Returns one row per (scan image, well) for wells that also have at least
    one LAS file. LAS URLs and depth ranges for the well are joined in so a
    scan row can later be matched to its LAS depth interval (pairs.py).
    """
    out_path = out_path or DATA_DIR / "manifest_pairs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("downloading LAS index ...")
    las = _fetch_index(LAS_INDEX_URL)
    print(f"  {len(las):,} LAS file entries")

    print("downloading scan index ...")
    scan = _fetch_index(SCAN_INDEX_URL)
    print(f"  {len(scan):,} scan entries")

    las = las[las["API_NUM_NODASH"].notna() & (las["API_NUM_NODASH"] != "")]
    scan = scan[scan["API_NUM_NODASH"].notna() & (scan["API_NUM_NODASH"] != "")]
    scan = scan[scan["SCAN_URL"].notna() & (scan["SCAN_URL"] != "")]

    paired_apis = set(las["API_NUM_NODASH"]) & set(scan["API_NUM_NODASH"])
    print(f"wells with both scan and LAS: {len(paired_apis):,}")

    scan_p = scan[scan["API_NUM_NODASH"].isin(paired_apis)].copy()
    scan_p["SCAN_URL_ORIG"] = scan_p["SCAN_URL"]
    scan_p["SCAN_URL"] = scan_p["SCAN_URL"].map(rewrite_url)

    las_p = las[las["API_NUM_NODASH"].isin(paired_apis)].copy()
    las_p["URL"] = las_p["URL"].map(rewrite_url)

    # aggregate the well's LAS files into one column: url|start|stop;url|start|stop
    las_p["las_entry"] = (
        las_p["URL"].fillna("")
        + "|" + las_p["Depth_start"].fillna("")
        + "|" + las_p["Depth_stop"].fillna("")
    )
    las_by_well = las_p.groupby("API_NUM_NODASH")["las_entry"].agg(";".join).rename("LAS_FILES")

    manifest = scan_p.merge(las_by_well, on="API_NUM_NODASH", how="inner")

    keep = [
        "API_NUM_NODASH", "API_NUMBER", "LEASE_AND_WELL", "OPERATOR", "FIELD",
        "LATITUDE", "LONGITUDE", "LOGGER", "TOOL", "LOG_TOP", "LOG_BOTTOM",
        "LOG_DATE", "SCAN_URL", "SCAN_URL_ORIG", "LAS_FILES",
    ]
    manifest = manifest[[c for c in keep if c in manifest.columns]]

    manifest.to_csv(out_path, index=False)
    print(f"wrote {len(manifest):,} scan rows for {manifest['API_NUM_NODASH'].nunique():,} "
          f"paired wells -> {out_path}")
    return manifest


if __name__ == "__main__":
    build_manifest()
