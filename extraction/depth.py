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
    depth_band: tuple | None = None  # (x_left, x_right) band that OCR'd

    def depth_at(self, row: float) -> float:
        return self.slope * row + self.intercept

    def row_at(self, depth: float) -> float:
        return (depth - self.intercept) / self.slope


def _parse_label(text: str) -> int | None:
    """Parse an OCR'd string as a depth label, or None if it isn't one."""
    text = text.replace(",", "").replace(" ", "")
    if not text.isdigit() or not 2 <= len(text) <= 5:
        return None
    value = int(text)
    # depth labels are round numbers; anything else is an OCR misread or a
    # stray mark (grid digits, specks) and poisons the fit
    if value < 50 or value % 25 != 0:
        return None
    return value


def _read_segment(seg: np.ndarray, upscale: float, min_conf: float,
                  rot: int) -> list[tuple[float, float]]:
    """OCR one segment at a given rotation (0, 1 or 3 quarter-turns).

    NOTE: easyocr's rotation_info option is broken in practice (it can
    replace a good horizontal read with edge junk), so rotated labels are
    handled by physically rotating the segment instead.
    Returns (row_in_segment, value) points mapped back to unrotated rows.
    """
    h = seg.shape[0]
    arr = np.rot90(seg, rot) if rot else seg
    im = Image.fromarray(arr)
    if upscale > 1.0:
        im = im.resize((int(im.width * upscale), int(im.height * upscale)),
                       Image.LANCZOS)
    out = []
    for box, text, conf in _reader().readtext(
            np.asarray(im), allowlist="0123456789,", detail=1,
            paragraph=False):
        if conf < min_conf:
            continue
        value = _parse_label(text)
        if value is None:
            continue
        cx = sum(p[0] for p in box) / len(box) / upscale
        cy = sum(p[1] for p in box) / len(box) / upscale
        if rot == 0:
            row = cy
        elif rot == 1:      # rotated CCW: original row = x in rotated frame
            row = cx
        else:               # rot == 3, rotated CW
            row = h - 1 - cx
        out.append((row, float(value)))
    return out


def _ocr_depth_labels(band: np.ndarray, dpi: float,
                      min_conf: float = 0.3) -> list[tuple[float, float]]:
    """OCR all depth labels in the depth-column band.

    Runs easyocr over the band in segments (the band can be 80k+ rows
    tall). Labels printed rotated 90 deg are handled by a second pass with
    rotated segments, run only when the horizontal pass reads too little.
    Returns (row_in_band, depth) points.
    """
    # upscale narrow/low-dpi bands so digits reach a size easyocr likes
    upscale = max(1.0, 120.0 / max(1.0, 0.12 * dpi))
    # easyocr shrinks images whose long side exceeds its canvas (~2560 px),
    # which destroys the digits; keep each segment under that AFTER upscaling
    seg_h = int(2300 / upscale)
    overlap = 200

    segments = []
    for seg_top in range(0, band.shape[0], seg_h - overlap):
        seg = band[seg_top:seg_top + seg_h]
        if seg.size and (seg < DARK_THRESHOLD).mean() >= 0.0005:
            segments.append((seg_top, seg))

    points: list[tuple[float, float]] = []

    def _collect(rot: int):
        for seg_top, seg in segments:
            for row_in_seg, value in _read_segment(seg, upscale, min_conf, rot):
                row = seg_top + row_in_seg
                # de-duplicate labels found twice in overlapping segments
                if any(abs(row - r) < 20 and value == d for r, d in points):
                    continue
                points.append((row, value))

    _collect(0)
    if len(points) < 3:  # horizontal pass too thin -> try rotated labels
        for rot in (1, 3):
            _collect(rot)
            if len(points) >= 3:
                break
    return points


def _consensus_fit(points: list[tuple[float, float]],
                   slope_bounds: tuple[float, float] | None = None
                   ) -> tuple[float, float, list]:
    """RANSAC line fit through (row, depth) points.

    OCR (especially the trained model on hard scans) yields a mix of good
    labels and repeated misreads; a median of pairwise slopes collapses to
    zero under duplicates. RANSAC instead tries every point pair as a line
    hypothesis, keeps only physically plausible slopes, and picks the line
    with the most inliers.
    """
    pts = sorted(points)
    if len(pts) < 3:
        raise ValueError("not enough labels for a slope estimate")

    lo, hi = slope_bounds if slope_bounds else (1e-6, np.inf)

    best_inliers: list = []
    best_rms = np.inf
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            (r1, d1), (r2, d2) = pts[i], pts[j]
            if r2 - r1 < 50 or d2 == d1:
                continue
            slope = (d2 - d1) / (r2 - r1)
            if not (lo <= slope <= hi):
                continue
            intercept = d1 - slope * r1
            tol = max(5.0, slope * 40)  # ~40 rows of drift allowed
            inliers = [(r, d) for r, d in pts
                       if abs(d - (slope * r + intercept)) <= tol]
            # score by DISTINCT depth values: a true depth scale passes many
            # different round numbers; junk lines are built from repeated
            # misreads of the same few values
            distinct = len({d for _, d in inliers})
            if distinct < 3:
                continue
            rms = float(np.sqrt(np.mean(
                [(d - (slope * r + intercept)) ** 2 for r, d in inliers])))
            best_distinct = len({d for _, d in best_inliers})
            if (distinct, -rms) > (best_distinct, -best_rms):
                best_inliers, best_rms = inliers, rms

    if len(best_inliers) < 3:
        raise ValueError(f"only {len(best_inliers)} labels agree with consensus")

    rows = np.array([r for r, _ in best_inliers])
    depths = np.array([d for _, d in best_inliers])
    A = np.vstack([rows, np.ones_like(rows)]).T
    (m, b), *_ = np.linalg.lstsq(A, depths, rcond=None)
    return float(m), float(b), best_inliers


def calibrate_depth(path: str, layout: Layout) -> DepthCalibration:
    candidates = layout.depth_col_candidates or (
        [layout.depth_col] if layout.depth_col else [])
    if not candidates:
        raise ValueError("layout has no depth column")

    gray = load_gray(path)
    dpi = layout.width / 8.25

    # scoring heuristics can rank the wrong band first, and OCR (especially
    # the trained model) can produce confident junk in a wrong band. So a
    # band only wins by producing labels that FIT A LINE: fit every
    # candidate and keep the best fit by distinct-inlier count.
    from . import digit_reader

    # physical slope bounds: depth increases downward and log scales run
    # ~1:120 to 1:1200, i.e. roughly 4-120 ft of depth per inch of paper
    slope_bounds = (4.0 / dpi, 120.0 / dpi)

    best = None
    last_err = "no depth labels OCR'd in any candidate band"
    for left, right in candidates[:3]:
        band = gray[layout.log_top:layout.log_bottom, left + 3:right - 2]
        points = [(layout.log_top + row, depth)
                  for row, depth in _ocr_depth_labels(band, dpi)]
        if digit_reader.available():
            for row, depth in digit_reader.read_labels(band, dpi):
                abs_row = layout.log_top + row
                if not any(abs(abs_row - r) < 20 and depth == d
                           for r, d in points):
                    points.append((abs_row, depth))
        if len(points) < 3:
            last_err = f"only {len(points)} depth labels OCR'd"
            continue
        try:
            slope, intercept, inliers = _consensus_fit(
                points, slope_bounds=slope_bounds)
        except ValueError as e:
            last_err = str(e)
            continue
        n_distinct = len({d for _, d in inliers})
        rms = float(np.sqrt(np.mean(
            [(d - (slope * r + intercept)) ** 2 for r, d in inliers])))
        key = (n_distinct, -rms)
        if best is None or key > best[0]:
            best = (key, slope, intercept, inliers, points, (left, right))

    if best is None:
        raise ValueError(last_err)
    _, slope, intercept, inliers, points, used_band = best

    ft_per_inch = slope * dpi
    if not (4.0 <= ft_per_inch <= 120.0):
        raise ValueError(f"implausible scale: {ft_per_inch:.1f} ft/inch")

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
        depth_band=used_band,
    )
