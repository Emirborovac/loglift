"""Layout detection: find the log section, track boundaries, and depth column.

Well log scans follow the API standard layout:

    +--------------------------------------+
    |  header (well info, curve scales)    |
    +---------+------+---------------------+
    | track 1 |depth | tracks 2-3          |
    | (GR/SP) | col  | (resistivity/poro)  |
    |  ~~~~   | 1250 |    ~~~~~            |
    +---------+------+---------------------+

Detection strategy (classic CV, no ML):

- The log section is where long vertical border lines run down the image.
- Track borders are columns whose dark-pixel fraction stays high over most
  of the log section's height.
- The depth column is the band between track borders with the LOWEST
  interior line density (it is white except for depth numbers).

Works on a downsampled copy for speed; returns full-resolution coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DOWNSAMPLE_WIDTH = 800          # analysis width in px
DARK_THRESHOLD = 128            # gray level below which a pixel is "ink"
VLINE_MIN_RUN_FRAC = 0.55       # a track border must span >=55% of log height


@dataclass
class Layout:
    """All coordinates in full-resolution pixels."""
    width: int
    height: int
    log_top: int                # first row of the gridded log section
    log_bottom: int             # last row of the log section
    track_borders: list[int]    # x of vertical track border lines
    depth_col: tuple[int, int]  # (x_left, x_right) of the depth column
    tracks: list[tuple[int, int]]  # (x_left, x_right) per curve track

    def to_dict(self) -> dict:
        return asdict(self)


def load_gray(path: str) -> np.ndarray:
    """Load a scan as a full-resolution grayscale numpy array (0=black)."""
    im = Image.open(path).convert("L")
    return np.asarray(im)


def _downsample(gray: np.ndarray, width: int = DOWNSAMPLE_WIDTH) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    scale = width / w
    im = Image.fromarray(gray).resize((width, max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(im), scale


def _find_log_sections(dark: np.ndarray, min_rows: int = 300,
                       merge_gap: int = 15) -> list[tuple[int, int]]:
    """Row ranges of candidate log sections (a strip may hold several).

    The log grid produces rows with many dark pixels at regular intervals.
    Headers/footers have irregular, sparser rows. Composite strips contain
    several gridded sections separated by header inserts, so we return every
    long-enough run of active rows, merging runs split by small gaps.
    """
    row_frac = dark.mean(axis=1)
    # smooth over ~50 rows to bridge white gaps between grid lines
    kernel = np.ones(51) / 51
    smooth = np.convolve(row_frac, kernel, mode="same")
    active = smooth > max(0.02, np.percentile(smooth, 40) * 0.5)

    runs, run_start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and run_start is None:
            run_start = i
        elif not a and run_start is not None:
            runs.append([run_start, i])
            run_start = None

    # merge runs separated by small gaps (fold lines, insert stickers)
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    return [(a, b) for a, b in merged if b - a >= min_rows]


def _slab_line_centers(slab: np.ndarray) -> list[int]:
    """Centers of near-solid vertical line columns within one slab."""
    col = slab.mean(axis=0)
    present = col > VLINE_MIN_RUN_FRAC
    centers, i, n = [], 0, len(present)
    while i < n:
        if present[i]:
            j = i
            while j < n and present[j]:
                j += 1
            centers.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return centers


def _find_track_borders(dark: np.ndarray, log_top: int, log_bottom: int,
                        n_slabs: int = 24, match_px: int = 10) -> list[int]:
    """Columns acting as vertical track borders within the log section.

    Scans are skewed: a border line drifts tens of pixels over the full
    height, but within one short slab it is a near-solid column. So we
    detect line centers per slab, then chain nearby centers across slabs
    (a simple tracker). Chains present in most slabs are borders; we return
    each chain's median x.
    """
    h = log_bottom - log_top
    slab_h = max(1, h // n_slabs)

    slab_centers = []
    for s in range(log_top, log_bottom - slab_h + 1, slab_h):
        slab_centers.append(_slab_line_centers(dark[s:s + slab_h]))
    n_slabs_used = len(slab_centers)
    if n_slabs_used == 0:
        return []

    # chain line centers across consecutive slabs; a chain survives a couple
    # of missed slabs (folds, tape, degraded ink briefly break a border)
    max_misses = 2
    chains: list[list[int]] = []            # x positions per chain
    active: list[list[int]] = []            # [chain_idx, last_x, misses]
    for centers in slab_centers:
        new_active = []
        unmatched = list(centers)
        for chain_idx, last_x, misses in active:
            best, best_d = None, match_px + 1
            for x in unmatched:
                d = abs(x - last_x)
                if d < best_d:
                    best, best_d = x, d
            if best is not None:
                unmatched.remove(best)
                chains[chain_idx].append(best)
                new_active.append([chain_idx, best, 0])
            elif misses < max_misses:
                new_active.append([chain_idx, last_x, misses + 1])
        for x in unmatched:
            chains.append([x])
            new_active.append([len(chains) - 1, x, 0])
        active = new_active

    borders = [int(np.median(c)) for c in chains if len(c) >= 0.6 * n_slabs_used]
    return sorted(borders)


def _find_depth_column(dark: np.ndarray, log_top: int, log_bottom: int,
                       borders: list[int]) -> tuple[int, int] | None:
    """Among bands between borders, pick the mostly-white one (depth column).

    The depth column has no grid: its interior dark fraction is far lower
    than curve tracks. Require a plausible width (2-15% of image width).
    """
    if len(borders) < 2:
        return None
    w = dark.shape[1]
    section = dark[log_top:log_bottom]

    best, best_score = None, 1.0
    for left, right in zip(borders, borders[1:]):
        band_w = right - left
        if not (0.02 * w <= band_w <= 0.15 * w):
            continue
        interior = section[:, left + 2:right - 1]
        if interior.size == 0:
            continue
        score = interior.mean()
        if score < best_score:
            best, best_score = (left, right), score
    # a real depth column is nearly white (few % ink from the numbers)
    if best is not None and best_score < 0.08:
        return best
    return None


def _analyze_section(dark: np.ndarray, log_top: int, log_bottom: int,
                     width: int) -> tuple[list[int], tuple | None, list]:
    borders = _find_track_borders(dark, log_top, log_bottom)
    depth_col = _find_depth_column(dark, log_top, log_bottom, borders)

    # tracks = bands between consecutive borders, excluding the depth column
    tracks = []
    min_track_w = 0.08 * width
    for left, right in zip(borders, borders[1:]):
        if depth_col is not None and (left, right) == depth_col:
            continue
        if right - left >= min_track_w:
            tracks.append((left, right))
    return borders, depth_col, tracks


def detect_layout(path: str) -> Layout:
    gray = load_gray(path)

    # section rows: coarse structure, downsampled analysis is fine.
    # NOTE: downsampling averages thin black lines into light gray, so
    # anything that depends on 1-3 px lines must use the FULL-RES mask.
    small, scale = _downsample(gray)
    dark_small = (small < DARK_THRESHOLD).astype(np.float32)
    sections = _find_log_sections(dark_small)

    # borders & depth column: full-resolution binary mask.
    # A strip may hold several log sections (main log, repeat section);
    # analyze each and keep the best-scoring one as the primary layout.
    dark = (gray < DARK_THRESHOLD).astype(np.float32)

    best = None
    for top_s, bottom_s in sections:
        log_top, log_bottom = int(top_s / scale), int(bottom_s / scale)
        borders, depth_col, tracks = _analyze_section(
            dark, log_top, log_bottom, gray.shape[1])
        # depth column is the strongest signal a section is a real log
        score = (10 if depth_col else 0) + len(tracks) \
            + 0.000001 * (log_bottom - log_top)
        if best is None or score > best[0]:
            best = (score, log_top, log_bottom, borders, depth_col, tracks)

    if best is None:
        h = gray.shape[0]
        best = (0, 0, h, [], None, [])

    _, log_top, log_bottom, borders, depth_col, tracks = best
    return Layout(
        width=gray.shape[1],
        height=gray.shape[0],
        log_top=log_top,
        log_bottom=log_bottom,
        track_borders=borders,
        depth_col=depth_col,
        tracks=tracks,
    )


def draw_layout(path: str, layout: Layout, out_path: str,
                thumb_width: int = 500) -> None:
    """Save a thumbnail with the detected layout drawn on it (debugging)."""
    from PIL import ImageDraw

    im = Image.open(path).convert("RGB")
    scale = thumb_width / im.width
    im = im.resize((thumb_width, int(im.height * scale)))
    d = ImageDraw.Draw(im)

    def x(v): return int(v * scale)
    def y(v): return int(v * scale)

    d.rectangle([0, y(layout.log_top), im.width - 1, y(layout.log_bottom)],
                outline=(0, 160, 255), width=3)
    for b in layout.track_borders:
        d.line([x(b), y(layout.log_top), x(b), y(layout.log_bottom)],
               fill=(255, 0, 0), width=2)
    if layout.depth_col:
        l, r = layout.depth_col
        d.rectangle([x(l), y(layout.log_top), x(r), y(layout.log_bottom)],
                    outline=(0, 200, 0), width=4)
    im.save(out_path)
