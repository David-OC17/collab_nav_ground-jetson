"""
arena_marker_localizer.quality
─────────────────────────────────────────────────────────────────────────────
Standalone per-frame quality filter. Drops frames that are too blurry or
that show heavy codec/compression artifacts. Modeled on the gating logic
in drone_map_grid_gen.ExtractionConfig but stripped to quality-only
(movement gates are excluded — irrelevant when the goal is marker
visibility, not stitching).

Two scores per frame:
  - Laplacian variance (higher = sharper)
  - DCT-block artifact ratio (higher = blockier / more codec damage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2


@dataclass
class QualityConfig:
    blur_thresh:     float = 60.0
    """Minimum Laplacian variance; frames below this are blurry."""
    artifact_thresh: float = 2.0
    """Maximum DCT block artifact ratio; frames above this are likely corrupt."""


def laplacian_variance(gray: np.ndarray) -> float:
    """Standard Laplacian-variance focus measure. Higher = sharper."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def dct_artifact_ratio(gray: np.ndarray, block: int = 8) -> float:
    """A cheap proxy for compression-block artifacts: ratio of
    energy along the 8-pixel block grid vs the off-grid energy in the
    same row/column. A clean frame is near 1.0; a frame riddled with
    JPEG/H.264 block boundaries climbs above 2-3."""
    h, w = gray.shape[:2]
    if h < block * 4 or w < block * 4:
        return 1.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)

    # Mean gradient on block boundaries vs in between.
    col_idx = np.arange(w)
    row_idx = np.arange(h)
    on_grid_cols  = (col_idx % block) == 0
    off_grid_cols = ~on_grid_cols
    on_grid_rows  = (row_idx % block) == 0
    off_grid_rows = ~on_grid_rows

    e_on  = (abs_gx[:, on_grid_cols].mean() + abs_gy[on_grid_rows, :].mean()) / 2.0
    e_off = (abs_gx[:, off_grid_cols].mean() + abs_gy[off_grid_rows, :].mean()) / 2.0
    if e_off < 1e-6:
        return 1.0
    return float(e_on / e_off)


def frame_passes(frame_bgr: np.ndarray, cfg: QualityConfig
                 ) -> Tuple[bool, float, float]:
    """Return (ok, blur_score, artifact_score).

    artifact_score is -1.0 when the blur gate already rejected the frame
    (the more expensive Sobel-based check is skipped in that case).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_variance(gray)
    if blur < cfg.blur_thresh:
        return False, blur, -1.0
    artifact = dct_artifact_ratio(gray)
    return artifact <= cfg.artifact_thresh, blur, artifact
