"""LAS export: write extracted curves to a standard LAS 2.0 file.

Takes the layout, depth calibration and traced curves and produces a file
any industry software (Petrel, Techlog, Kingdom, lasio...) can read.

Curves without header-derived units yet are exported normalized 0..1 with
mnemonics TRK<n>_C<m> and a comment explaining the convention; once curve
identification lands they get real names (GR, NPHI...) and units.
"""

from __future__ import annotations

import numpy as np
import lasio

from extraction.layout import Layout
from extraction.depth import DepthCalibration


def write_las(path: str, layout: Layout, cal: DepthCalibration,
              track_curves: dict[int, list[np.ndarray]],
              well_name: str = "", api: str = "",
              step_ft: float = 0.5, source_scan: str = "") -> str:
    """Write traced curves to a LAS 2.0 file resampled on a regular grid.

    track_curves: {track_index: [curve arrays (normalized 0..1) per row]}
    Rows run layout.log_top..layout.log_bottom; depths come from `cal`.
    """
    rows = np.arange(layout.log_top, layout.log_bottom)
    depths = cal.slope * rows + cal.intercept

    top = np.ceil(depths.min() / step_ft) * step_ft
    bot = np.floor(depths.max() / step_ft) * step_ft
    grid = np.arange(top, bot + step_ft / 2, step_ft)

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value=well_name)
    las.well["API"] = lasio.HeaderItem("API", value=api)
    las.well["STRT"] = lasio.HeaderItem("STRT", unit="FT", value=float(grid[0]))
    las.well["STOP"] = lasio.HeaderItem("STOP", unit="FT", value=float(grid[-1]))
    las.well["STEP"] = lasio.HeaderItem("STEP", unit="FT", value=step_ft)
    las.well["NULL"] = lasio.HeaderItem("NULL", value=-999.25)
    las.other = (
        "Digitized from raster scan by LogLift. "
        f"Source: {source_scan}. "
        f"Depth calibration: {cal.n_inliers} labels, "
        f"RMS {cal.rms_residual_ft:.2f} ft. "
        "Curve values are track-normalized (0=left edge, 1=right edge) "
        "until unit calibration is applied."
    )

    las.append_curve("DEPT", grid, unit="FT")

    # depths can decrease with row on inverted scans; np.interp needs
    # ascending x, so sort once
    order = np.argsort(depths)
    d_sorted = depths[order]

    for t_idx, curves in sorted(track_curves.items()):
        for c_idx, curve in enumerate(curves):
            c_sorted = np.asarray(curve)[order]
            resampled = np.interp(grid, d_sorted, c_sorted,
                                  left=np.nan, right=np.nan)
            las.append_curve(
                f"TRK{t_idx}_C{c_idx}", resampled, unit="norm",
                descr=f"track {t_idx} curve {c_idx} (normalized position)")

    with open(path, "w") as f:
        las.write(f, version=2.0)
    return path
