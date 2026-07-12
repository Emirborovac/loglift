"""Custom depth-label reader: our trained CRNN, with graceful absence.

If data/models/digit_crnn.pt exists, calibrate_depth runs it on candidate
label blobs IN ADDITION to easyocr — the consensus fit dedupes and
arbitrates. If the model file is missing, everything behaves as before.

The CRNN is a recognizer, not a detector, so blobs are located first with
the sparse-ink heuristic (labels are ink islands in a blank column).
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..",
                          "data", "models", "digit_crnn.pt")

_MODEL = None
_AVAILABLE = None


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = os.path.exists(MODEL_PATH)
    return _AVAILABLE


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from training.train_digits import CRNN
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        m = CRNN()
        m.load_state_dict(ckpt["state"])
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _MODEL = m
    return _MODEL


def find_label_blobs(band: np.ndarray, dpi: float,
                     dark_threshold: int = 128) -> list[tuple[int, int]]:
    """Row ranges of candidate label blobs (ink islands) in the band."""
    min_h, max_h = int(0.04 * dpi), int(0.70 * dpi)
    dark = band < dark_threshold
    row_has_ink = dark.mean(axis=1) > 0.02

    blobs, start = [], None
    pad = int(0.03 * dpi)
    for i, a in enumerate(np.append(row_has_ink, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if min_h <= i - start <= max_h:
                blobs.append((max(0, start - pad),
                              min(band.shape[0], i + pad)))
            start = None
    return blobs


def read_blob(band: np.ndarray, top: int, bottom: int,
              min_conf: float = 0.85) -> int | None:
    """CRNN-read one blob; returns the integer value or None."""
    import torch
    from training.train_digits import CHARS, BLANK, IMG_H, ctc_decode

    crop = band[top:bottom]
    if crop.size == 0:
        return None
    im = Image.fromarray(crop)
    w = max(8, int(im.width * IMG_H / im.height))
    im = im.resize((min(w, 256), IMG_H), Image.BILINEAR)
    x = torch.from_numpy(
        np.asarray(im, dtype=np.float32)[None, None] / 255.0)

    m = _model()
    if next(m.parameters()).is_cuda:
        x = x.cuda()
    with torch.no_grad():
        logits = m(x)[0]

    text = ctc_decode(logits)
    if not text.isdigit() or not 2 <= len(text) <= 5:
        return None
    # pseudo-confidence: mean prob of the argmax path over non-blank steps
    probs = logits.softmax(-1)
    ids = logits.argmax(-1)
    mask = ids != BLANK
    if not bool(mask.any()):
        return None
    conf = float(probs.max(-1).values[mask].mean())
    if conf < min_conf:
        return None
    return int(text)


def read_labels(band: np.ndarray, dpi: float) -> list[tuple[float, float]]:
    """All (row_in_band, value) label readings from the trained model."""
    if not available():
        return []
    points = []
    for top, bottom in find_label_blobs(band, dpi):
        value = read_blob(band, top, bottom)
        if value is not None and value >= 50 and value % 25 == 0:
            points.append(((top + bottom) / 2, float(value)))
    return points
