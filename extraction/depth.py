"""Depth calibration: map image rows to depth in feet.

The depth column contains printed depth labels (e.g. 450, 500, 550) at
regular spacing. We locate the label blobs, OCR them (easyocr, digits only),
then fit depth = slope * row + intercept robustly:

- labels must be monotonic in depth (increasing downward, occasionally up)
- spacing is regular, so pairwise slopes cluster tightly; outlier OCR
  readings (misread digits) are rejected by consensus before the final fit

Returns the calibration plus quality metrics so downstream code can refuse
low-confidence scans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .layout import Layout, load_gray, DARK_THRESHOLD

Image.MAX_IMAGE_PIXELS = None

_READER = None  # lazy singleton; easyocr loads a model


def _reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], verbose=False)
    return _READER


@dataclass
class DepthCalibration:
    slope: float            # feet per pixel row
    intercept: float        # depth at row 0
    n_labels: int           # OCR'd labels used in the fit
    n_inliers: int          # labels surviving consensus
    rms_residual_ft: float  # fit quality
    labels: list            # (row, depth) inliers, for debugging

    def depth_at(self, row: float) -> float:
        return self.slope * row + self.intercept

    def row_at(self, depth: float) -> float:
        return (depth - self.intercept) / self.slope


def _find_label_blobs(band: np.ndarray, min_h: int = 15, max_h: int = 120,
                      pad: int = 8) -> list[tuple[int, int]]:
    """Row ranges of candidate depth-label blobs inside the depth column.

    The column is white except for the printed numbers, so blobs are runs of
    rows containing ink.
    """
    dark = band < DARK_THRESHOLD
    row_has_ink = dark.mean(axis=1) > 0.02

    blobs, start = [], None
    for i, a in enumerate(np.append(row_has_ink, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            h = i - start
            if min_h <= h <= max_h:
                blobs.append((max(0, start - pad), min(band.shape[0], i + pad)))
            start = None
    return blobs


def _ocr_blob(band: np.ndarray, top: int, bottom: int) -> int | None:
    """OCR one label blob; returns the depth number or None."""
    crop = band[top:bottom]
    # upscale small crops for better OCR
    im = Image.fromarray(crop)
    scale = max(1, int(60 / max(1, im.height)))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    result = _reader().readtext(np.asarray(im), allowlist="0123456789",
                                detail=1, paragraph=False)
    if not result:
        return None
    # take the highest-confidence reading
    _, text, conf = max(result, key=lambda r: r[2])
    if conf < 0.4 or not text.isdigit() or not 1 <= len(text) <= 5:
        return None
    return int(text)


def _consensus_fit(points: list[tuple[float, float]]) -> tuple[float, float, list]:
    """Robust line fit through (row, depth) points.

    Depth labels are evenly spaced, so the true slope is the strong mode of
    pairwise slopes. Points agreeing with the consensus line are inliers.
    """
    pts = sorted(points)
    slopes = []
    for i in range(len(pts) - 1):
        (r1, d1), (r2, d2) = pts[i], pts[i + 1]
        if r2 - r1 > 10:
            slopes.append((d2 - d1) / (r2 - r1))
    if not slopes:
        raise ValueError("not enough labels for a slope estimate")
    slope = float(np.median(slopes))

    # intercept consensus with the median slope
    intercepts = [d - slope * r for r, d in pts]
    intercept = float(np.median(intercepts))

    # inliers: within a tolerance of the consensus line
    tol = max(5.0, abs(slope) * 40)  # ~40 rows of drift allowed
    inliers = [(r, d) for r, d in pts if abs(d - (slope * r + intercept)) <= tol]
    if len(inliers) < 3:
        raise ValueError(f"only {len(inliers)} labels agree with consensus")

    # final least-squares fit on inliers
    rows = np.array([r for r, _ in inliers])
    depths = np.array([d for _, d in inliers])
    A = np.vstack([rows, np.ones_like(rows)]).T
    (m, b), *_ = np.linalg.lstsq(A, depths, rcond=None)
    return float(m), float(b), inliers


def calibrate_depth(path: str, layout: Layout) -> DepthCalibration:
    if layout.depth_col is None:
        raise ValueError("layout has no depth column")

    gray = load_gray(path)
    left, right = layout.depth_col
    band = gray[layout.log_top:layout.log_bottom, left + 3:right - 2]

    blobs = _find_label_blobs(band)
    points = []
    for top, bottom in blobs:
        value = _ocr_blob(band, top, bottom)
        if value is not None:
            row_center = layout.log_top + (top + bottom) / 2
            points.append((row_center, float(value)))

    if len(points) < 3:
        raise ValueError(f"only {len(points)} depth labels OCR'd")

    slope, intercept, inliers = _consensus_fit(points)

    rows = np.array([r for r, _ in inliers])
    depths = np.array([d for _, d in inliers])
    rms = float(np.sqrt(np.mean((slope * rows + intercept - depths) ** 2)))

    return DepthCalibration(
        slope=slope,
        intercept=intercept,
        n_labels=len(points),
        n_inliers=len(inliers),
        rms_residual_ft=rms,
        labels=inliers,
    )
