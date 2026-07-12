"""Header reading: curve names and scale endpoints per track.

The scale block above the log grid announces each track's curves and
engineering scales, e.g.:

    GAMMA RAY API UNITS          BULK DENSITY Pb-gm/cc
    0            150             2.0        2.5        3.0
    Caliper hole diameter...     POROSITY %
                                 30    20    10    0    -10

We OCR the region above the log section and:
1. match curve-family keywords (GAMMA RAY -> GR, BULK DENSITY -> RHOB...)
2. collect numeric tokens lying inside each track's x-range and group them
   into scale lines (same text row); a line's leftmost/rightmost numbers
   are the track edge values

Output feeds unit scaling: value = left + norm * (right - left).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .layout import Layout, load_gray

Image.MAX_IMAGE_PIXELS = None

# keyword -> (mnemonic, unit)
CURVE_KEYWORDS = [
    ("GAMMA RAY", "GR", "GAPI"),
    ("GAMMA", "GR", "GAPI"),
    ("CALIPER", "CALI", "IN"),
    ("HOLE DIAMETER", "CALI", "IN"),
    ("BULK DENSITY", "RHOB", "G/C3"),
    ("PB-GM", "RHOB", "G/C3"),        # 'Pb-gm/cc' unit line names the curve
    ("DENSITY CORRECTION", "DRHO", "G/C3"),
    ("POROSITY", "PHI", "PU"),
    ("NEUTRON", "NPHI", "PU"),
    ("RESISTIVITY", "RES", "OHMM"),
    ("INDUCTION", "ILD", "OHMM"),
    ("CONDUCTIVITY", "COND", "MMHO"),
    ("TRANSIT TIME", "DT", "US/F"),
    ("SONIC", "DT", "US/F"),
    ("SP ", "SP", "MV"),
    ("SELF POTENTIAL", "SP", "MV"),
    ("TENSION", "TENS", "LBS"),
]

_NUM = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class TrackScale:
    track: int              # track index in layout.tracks
    mnemonic: str
    unit: str
    left_value: float | None = None
    right_value: float | None = None


def _ocr_region(gray: np.ndarray, y0: int, y1: int, reader) -> list:
    """OCR rows y0..y1 in canvas-sized slices; returns (cx, cy, text, conf)."""
    out = []
    seg_h = 2300
    for top in range(y0, y1, seg_h - 150):
        seg = gray[top:min(top + seg_h, y1)]
        if seg.size == 0 or (seg < 128).mean() < 0.001:
            continue
        for box, text, conf in reader.readtext(seg, detail=1, paragraph=False):
            if conf < 0.3 or not text.strip():
                continue
            cx = sum(p[0] for p in box) / len(box)
            cy = top + sum(p[1] for p in box) / len(box)
            out.append((cx, cy, text.strip(), conf))
    return out


def _scale_rows(tokens: list, x_left: int, x_right: int,
                row_tol: float) -> list[tuple[float, float, float]]:
    """Scale lines in a track: rows of >=2 numbers spanning most of it.

    Returns (y, left_value, right_value) per scale line.
    """
    nums = [(cx, cy, float(t.replace(",", ""))) for cx, cy, t, _ in tokens
            if _NUM.match(t.replace(",", "")) and x_left <= cx <= x_right]
    nums.sort(key=lambda n: n[1])

    rows: list[list] = []
    for n in nums:
        if rows and abs(n[1] - rows[-1][-1][1]) <= row_tol:
            rows[-1].append(n)
        else:
            rows.append([n])

    out = []
    for row in rows:
        if len(row) < 2:
            continue
        spread = max(n[0] for n in row) - min(n[0] for n in row)
        if spread < 0.5 * (x_right - x_left):
            continue  # clustered numbers, not a full scale line
        row.sort(key=lambda n: n[0])
        y = float(np.mean([n[1] for n in row]))
        out.append((y, row[0][2], row[-1][2]))
    return out


def read_header(path: str, layout: Layout, reader=None,
                scale_zone_in: float = 6.0) -> list[TrackScale]:
    """Curve names + scale endpoints for each track from the header.

    Only the scale zone (the last few inches above the log grid) is used:
    the well-info form higher up is full of small numbers that fake scale
    lines. Each scale line is assigned to the keyword directly above it.
    """
    if reader is None:
        from .depth import _reader
        reader = _reader()

    gray = load_gray(path)
    dpi = layout.width / 8.25
    y0 = max(0, layout.log_top - int(scale_zone_in * dpi))
    tokens = [(cx, cy, t.upper(), c) for cx, cy, t, c in
              _ocr_region(gray, y0, layout.log_top, reader)]

    row_tol = 0.12 * dpi
    max_kw_gap = 1.2 * dpi   # scale line must sit within ~1.2in below its name

    scales = []
    for t_idx, (x_left, x_right) in enumerate(layout.tracks):
        keywords = []   # (cy, mnemonic, unit)
        for cx, cy, text, conf in tokens:
            if not (x_left <= cx <= x_right):
                continue
            for kw, mnem, unit in CURVE_KEYWORDS:
                if kw in text and all(m != mnem for _, m, _ in keywords):
                    keywords.append((cy, mnem, unit))
                    break

        rows = _scale_rows(tokens, x_left, x_right, row_tol)
        used_rows = set()
        for kw_y, mnem, unit in sorted(keywords):
            ts = TrackScale(t_idx, mnem, unit)
            # nearest unclaimed scale line below the keyword
            best_i, best_gap = None, max_kw_gap
            for i, (row_y, lv, rv) in enumerate(rows):
                gap = row_y - kw_y
                if i not in used_rows and 0 <= gap <= best_gap:
                    best_i, best_gap = i, gap
            if best_i is not None:
                used_rows.add(best_i)
                _, ts.left_value, ts.right_value = rows[best_i]
            scales.append(ts)
    return scales
