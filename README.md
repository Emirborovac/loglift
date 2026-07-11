# LogLift

**AI that converts scanned paper well logs (TIFF/PDF) into digital LAS files.**

## The problem

Millions of oil and gas wells were logged before the digital era. Their measurements exist only as scanned images of paper strips — unreadable by any modern software. One commercial archive alone holds 7+ million of these raster logs across 3 million wells.

Today, converting a raster log to usable digital data means either manual curve tracing in 30-year-old desktop tools, or paying a service company per log. Meanwhile, demand for this old data is growing fast: well re-development, CO₂ storage site screening, and geothermal well repurposing all start with legacy well records.

## The solution

LogLift reads a scanned well log image and produces a calibrated digital LAS file:

1. **Layout detection** — find the log tracks, depth column, and header on the image
2. **Grid & depth calibration** — read the depth numbers and grid lines to map pixels to depth and scale
3. **Curve extraction** — trace each curve through the track, separating overlapping curves by line style
4. **Unit scaling** — convert pixel positions to real engineering units (gAPI, ohm·m, g/cm³ ...)
5. **Validation** — check the output against physical rules (value ranges, curve consistency) before export
6. **LAS export** — write a standard LAS 2.0 file any industry software can read

## Why it is feasible

The Kansas Geological Survey (KGS) publishes both the scanned image **and** the verified digital LAS file for **22,490 of the same wells** (~93,000 log images) — free, no registration. That gives us ground-truth training and testing pairs at a scale that did not exist when today's digitizing tools were built.

Verified (July 2026):

| Check | Result |
|---|---|
| Kansas wells with a digital LAS file | 24,029 |
| Kansas wells with scanned paper logs | 140,694 |
| **Wells with both (training pairs)** | **22,490** |
| Scan images for those paired wells | 93,567 TIFFs |
| Access | Free, no login |

More states (North Dakota, Oklahoma, Texas) can be added with the same approach later.

## Project structure

```
loglift/
├── pipeline/          # data pipeline: build the paired scan+LAS dataset from KGS
│   ├── indexes.py     #   download & join the KGS scan/LAS indexes
│   ├── download.py    #   fetch paired TIFF + LAS files
│   └── pairs.py       #   match each scan image to its LAS depth interval
├── extraction/        # image → curves (the core model / CV work)
├── validation/        # petrophysical sanity checks on extracted curves
├── export/            # LAS file writing
└── data/              # local data (gitignored)
```

## Status

- [x] Problem researched and data availability verified
- [x] KGS index pipeline (join scans ↔ LAS by API number)
- [ ] Paired dataset builder (download + scan-to-LAS matching)
- [ ] Track/grid layout detection
- [ ] Curve extraction
- [ ] Validation & LAS export
- [ ] Benchmark against ground truth

## Data sources

- [KGS LAS files database](https://www.kgs.ku.edu/Magellan/Logs/index.html)
- [KGS scanned wireline logs](https://www.kgs.ku.edu/Magellan/Elog/index.html)

Well data courtesy of the Kansas Geological Survey.
