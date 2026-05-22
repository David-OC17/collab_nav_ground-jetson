"""
arena_map_builder.consistency
─────────────────────────────────────────────────────────────────────────────
Self-consistency estimation for the transferred obstacle map.

We don't have ground truth, so per-obstacle "accuracy" is approximated by
combining three independent proxies, all in [0, 1]:

  A. bbox-mode ↔ grid-mode centroid agreement
     The transfer pipeline can project the same source blobs onto the clean
     background using two algorithmically independent mappings:
       * bbox  : normalize cleaned-image extent → background wall bbox
       * grid  : piecewise-linear via detected blue grid intersections
     If both modes place an obstacle within a fraction of a cell, the
     position is solid; if they disagree by more than a cell, something
     drifted.

  B. bbox-mode ↔ grid-mode area-ratio agreement
     The drawn obstacle's area should be roughly equal in both modes.
     The ratio min/max in {bbox_area, grid_area} is taken as a score
     (1.0 = identical, 0.0 = one is 0).

  C. Parameter-perturbation stability (close_iters ± 1)
     Re-run blob extraction with close_iters - 1 and close_iters + 1. For
     each blob we track:
       * centroid stability (std of centroid normalized to cell width)
       * area stability (coefficient of variation)
     Low variation → stable detection → high score.

The three scores are combined into a single per-obstacle confidence in
[0, 1] with configurable weights.

This module deliberately depends on the vendored ``processing`` API
(``TransferConfig``, ``run_pipeline``, ``Blob``) — it never reaches into
internals or modifies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import math
import numpy as np
import cv2

from .processing.transfer_obstacles import (
    TransferConfig,
    ExpectedShape,
    Blob,
    run_pipeline,
)


# ───────────────────────────────────────────────────────────────────────────
# Grid-safe default colors
# ───────────────────────────────────────────────────────────────────────────
# The vendored transfer_obstacles defaults pick "blue-ish" for boxes, but
# in BGR (200, 120, 40) that has the *same hue* (~108 in OpenCV's H) as
# the blue grid lines on the background template. Downstream code in this
# wrapper detects obstacles on the final composited image by color, so we
# need a color that's clearly outside the grid's hue band.
#
# Cyan-ish (255, 200, 0) BGR sits at hue ~90 — far enough from the grid
# that there's no ambiguity.

GRID_SAFE_SHAPES: List[ExpectedShape] = [
    ExpectedShape(
        name="box",
        descriptions=[
            "a cardboard box",
            "a rectangular box seen from above",
            "a square box",
        ],
        draw_color_bgr=(255, 200, 0),   # cyan-ish BGR — disjoint from blue grid
    ),
    ExpectedShape(
        name="cone",
        descriptions=[
            "a traffic cone",
            "an orange safety cone",
            "a cone seen from above",
        ],
        draw_color_bgr=(0, 140, 255),   # orange BGR
    ),
]


def _ensure_grid_safe_colors(cfg: TransferConfig) -> TransferConfig:
    """If the caller is using the vendored default `expected_shapes`
    (which collide with the blue grid in hue), substitute the
    grid-safe set. If the caller provided custom shapes, respect them."""
    from .processing.transfer_obstacles import _default_shapes
    vendored_defaults = _default_shapes()
    using_defaults = (
        len(cfg.expected_shapes) == len(vendored_defaults)
        and all(
            a.name == b.name and a.draw_color_bgr == b.draw_color_bgr
            for a, b in zip(cfg.expected_shapes, vendored_defaults)
        )
    )
    if using_defaults:
        return replace(cfg, expected_shapes=GRID_SAFE_SHAPES)
    return cfg


# ───────────────────────────────────────────────────────────────────────────
# Result containers
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ObstacleConsistency:
    """Per-obstacle confidence breakdown (each component is in [0, 1]).

    The `contour_px` is the obstacle's contour in pixel coordinates of the
    *final composited* (bbox-mode) image, so the rasterizer downstream can
    use it directly.
    """
    index:               int
    label:               str
    contour_px:          np.ndarray            # (N, 1, 2) int32
    centroid_px:         Tuple[float, float]
    area_px:             float
    bbox_grid_position:  float = 0.0           # proxy A
    bbox_grid_area:      float = 0.0           # proxy B
    perturbation:        float = 0.0           # proxy C
    overall:             float = 0.0           # weighted combination


@dataclass
class ConsistencyConfig:
    """How heavily to weight each proxy when combining into `overall`.
    Weights are renormalized so they always sum to 1.0."""
    w_bbox_grid_position: float = 0.45
    w_bbox_grid_area:     float = 0.25
    w_perturbation:       float = 0.30

    # Tunables for converting raw measurements to [0, 1].
    # Cell width in pixels is auto-detected from the background; these are
    # tolerances RELATIVE to that cell width.
    position_tol_cells: float = 0.5
    """Centroid disagreement at which the position score drops to ~0.
    Below ~0.1 cells you get ~1.0; above this it asymptotes to 0."""

    perturbation_stability_tol: float = 0.25
    """Combined CoV(area) + normalized_std(centroid) above which the
    perturbation score is ~0."""

    # Used by the perturbation pass.
    close_iters_perturbation: Tuple[int, int] = (-1, +1)
    """Offsets from cfg.close_iterations to use in the two extra passes."""


# ───────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────────────────────

def _centroid(blob: Blob) -> Tuple[float, float]:
    return blob.centroid


def _match_blobs_by_centroid(
    base: List[Blob],
    other: List[Blob],
    max_dist_px: float,
) -> List[Optional[int]]:
    """Greedy nearest-neighbour match. For each blob in `base` returns the
    index in `other` it matched to (or None). Each `other` index is used
    at most once."""
    if not base or not other:
        return [None] * len(base)

    used = set()
    out: List[Optional[int]] = []
    for b in base:
        bx, by = _centroid(b)
        best_i, best_d = None, math.inf
        for j, o in enumerate(other):
            if j in used:
                continue
            ox, oy = _centroid(o)
            d = math.hypot(bx - ox, by - oy)
            if d < best_d:
                best_d, best_i = d, j
        if best_i is not None and best_d <= max_dist_px:
            used.add(best_i)
            out.append(best_i)
        else:
            out.append(None)
    return out


def _estimate_cell_width_px(stages: dict) -> float:
    """Pull the median blue-grid cell width from the cleaned image.
    Falls back to a fraction of the image width when the grid can't be
    detected."""
    cleaned = stages.get("wall_masked", stages.get("cropped", stages["input"]))
    H, W = cleaned.shape[:2]

    # The transfer_obstacles internal grid detector returns sorted peaks,
    # we replicate a lightweight version here so we don't import private
    # helpers.
    import cv2
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 60, 60]), np.array([135, 255, 255]))
    col_counts = (mask > 0).sum(axis=0)
    thr = max(8, int(H * 0.30))

    peaks: List[int] = []
    in_peak = False
    start = 0
    for i, v in enumerate(col_counts):
        if v >= thr and not in_peak:
            in_peak, start = True, i
        elif v < thr and in_peak:
            in_peak = False
            peaks.append((start + i - 1) // 2)
    if in_peak:
        peaks.append((start + len(col_counts) - 1) // 2)

    if len(peaks) >= 2:
        diffs = np.diff(peaks)
        return float(np.median(diffs))
    return float(W) / 8.0  # safe fallback for an 8-cell arena


def _extract_drawn_contours(
    final_bgr: np.ndarray,
    base_cfg:  TransferConfig,
) -> dict:
    """For each expected shape (by label), return the list of contours
    drawn on `final_bgr` matching that shape's color. The transfer
    pipeline draws each obstacle filled in `shape.draw_color_bgr`; we
    threshold on that exact color with a small tolerance to recover the
    contours in the FINAL image's coordinate frame."""
    out: dict = {}
    for shape in base_cfg.expected_shapes:
        b, g, r = shape.draw_color_bgr
        tol = 25
        lo = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)],
                      dtype=np.uint8)
        hi = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)],
                      dtype=np.uint8)
        mask = cv2.inRange(final_bgr, lo, hi)
        if mask.sum() == 0:
            out[shape.name] = []
            continue
        # The outline drawing in transfer_obstacles uses a darker shade —
        # close to fuse outline + fill.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        out[shape.name] = list(contours)
    return out


def _match_drawn_contour(
    drawn_by_label: dict,
    label:          str,
    source_blob:    Blob,
    source_shape:   Tuple[int, int],   # (H, W) of cleaned source image
    final_shape:    Tuple[int, int],   # (H, W) of final composited image
) -> Optional[np.ndarray]:
    """Pick the drawn contour in `final_bgr` that best matches the
    source blob — using normalized centroid (independent of resolution)
    plus an area-ratio bonus."""
    candidates = drawn_by_label.get(label, [])
    if not candidates:
        return None

    Hs, Ws = source_shape
    Hf, Wf = final_shape
    sx, sy = source_blob.centroid
    src_nx = sx / max(Ws, 1)
    src_ny = sy / max(Hs, 1)
    src_area_frac = source_blob.area / max(Hs * Ws, 1)

    best, best_score = None, math.inf
    for c in candidates:
        a = cv2.contourArea(c)
        if a <= 0:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_f, cy_f = M["m10"] / M["m00"], M["m01"] / M["m00"]
        nx_f, ny_f = cx_f / max(Wf, 1), cy_f / max(Hf, 1)
        cand_area_frac = a / max(Hf * Wf, 1)

        d_pos  = math.hypot(src_nx - nx_f, src_ny - ny_f)
        # Asymmetric: positions weigh more than area mismatch.
        d_area = abs(src_area_frac - cand_area_frac) / max(src_area_frac, 1e-9)
        score = d_pos + 0.3 * d_area
        if score < best_score:
            best_score, best = score, c
    # Reject very poor matches.
    if best_score > 0.35:
        return None
    return best


# ───────────────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────────────

def compute_consistency(
    reconstructed_path: str,
    background_path:    str,
    base_cfg:           TransferConfig,
    cons_cfg:           Optional[ConsistencyConfig] = None,
    verbose:            bool = False,
    progress_cb=None,
) -> Tuple[np.ndarray, List[ObstacleConsistency], dict]:
    """Run the transfer pipeline three times and combine into per-obstacle
    confidences.

    Returns
    -------
    final_bgr   : np.ndarray
        The bbox-mode final composited image (used as the reference frame
        for everything downstream). HxWx3 BGR uint8.
    obstacles   : List[ObstacleConsistency]
        One entry per accepted bbox-mode obstacle, with per-proxy and
        overall scores filled in.
    stages_bbox : dict
        The full stages dict from the bbox-mode pass (so the caller has
        the cleaned image, masks, etc., for downstream debug).
    """
    cons_cfg = cons_cfg or ConsistencyConfig()
    base_cfg = _ensure_grid_safe_colors(base_cfg)

    def _emit(stage: str, frac: float, msg: str = ""):
        if progress_cb is not None:
            progress_cb(stage, frac, msg)

    # ── Pass 1: bbox mode (the reference projection) ────────────────────
    _emit("consistency", 0.05, "Pass 1/3: bbox-mode projection")
    cfg_bbox = replace(base_cfg, project_mode="bbox")
    final_bbox, stages_bbox = run_pipeline(
        reconstructed_path, background_path,
        cfg=cfg_bbox, verbose=verbose,
    )

    # ── Pass 2: grid mode (independent projection) ──────────────────────
    _emit("consistency", 0.40, "Pass 2/3: grid-mode projection")
    cfg_grid = replace(base_cfg, project_mode="grid")
    _final_grid, stages_grid = run_pipeline(
        reconstructed_path, background_path,
        cfg=cfg_grid, verbose=verbose,
    )

    # ── Pass 3 + 4: perturbation passes ─────────────────────────────────
    base_iters = base_cfg.close_iterations
    perturb_iters = [
        max(0, base_iters + cons_cfg.close_iters_perturbation[0]),
        base_iters + cons_cfg.close_iters_perturbation[1],
    ]
    perturb_blobs: List[List[Blob]] = []
    for k, it in enumerate(perturb_iters):
        _emit("consistency", 0.60 + 0.15 * k,
              f"Pass {3 + k}/4: perturbation close_iters={it}")
        cfg_p = replace(base_cfg, project_mode="bbox", close_iterations=it)
        _f, stages_p = run_pipeline(
            reconstructed_path, background_path,
            cfg=cfg_p, verbose=verbose,
        )
        # Re-use stages_p's blob overlay rather than re-extracting — but
        # the stages dict only stores the overlay image, not the Blob
        # objects themselves. We need them, so do one more extraction
        # against the perturbed cleaned mask.
        from .processing.transfer_obstacles import extract_blobs, build_clean_masks
        pink, _green = build_clean_masks(stages_p["wall_masked"], cfg_p,
                                         verbose=False)
        perturb_blobs.append(extract_blobs(pink, cfg_p, verbose=False))

    # ── Recover Blob lists from bbox and grid passes the same way ──────
    from .processing.transfer_obstacles import extract_blobs, build_clean_masks
    pink_bbox, _ = build_clean_masks(stages_bbox["wall_masked"], cfg_bbox,
                                     verbose=False)
    blobs_bbox = extract_blobs(pink_bbox, cfg_bbox, verbose=False)
    pink_grid, _ = build_clean_masks(stages_grid["wall_masked"], cfg_grid,
                                     verbose=False)
    blobs_grid = extract_blobs(pink_grid, cfg_grid, verbose=False)

    # Filter to only the obstacles that were actually drawn in bbox mode
    # (i.e. not 'unknown' and not dropped by drop_unknown).
    drawn_bbox = [
        b for b in blobs_bbox
        if (b.final_label and b.final_label != "unknown")
           or (not base_cfg.drop_unknown)
    ]

    # ── Build the per-obstacle records and score them ───────────────────
    cell_w = _estimate_cell_width_px(stages_bbox)
    max_match_dist = cell_w * 1.5   # be lenient when matching across passes

    grid_match = _match_blobs_by_centroid(drawn_bbox, blobs_grid, max_match_dist)
    perturb_matches = [
        _match_blobs_by_centroid(drawn_bbox, pl, max_match_dist)
        for pl in perturb_blobs
    ]

    # Re-detect contours on the FINAL composited image, where each
    # accepted obstacle is drawn with its shape's draw_color_bgr. The
    # source-image contour coordinates from the Blob objects can't be
    # used directly downstream because the rasterizer operates in
    # final_bbox-pixel space (background.png-sized).
    final_contours_by_label = _extract_drawn_contours(final_bbox, base_cfg)
    src_H, src_W = stages_bbox["wall_masked"].shape[:2]

    obstacles: List[ObstacleConsistency] = []
    for i, b in enumerate(drawn_bbox):
        label = b.final_label or "unknown"
        contour_in_final = _match_drawn_contour(
            final_contours_by_label, label, b,
            source_shape=(src_H, src_W),
            final_shape=final_bbox.shape[:2],
        )
        if contour_in_final is None:
            # Fall back to the source contour; downstream raster won't
            # render this cleanly but at least it's not lost.
            contour_in_final = b.contour
            cx, cy = b.centroid
        else:
            M = cv2.moments(contour_in_final)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = b.centroid

        rec = ObstacleConsistency(
            index=i,
            label=label,
            contour_px=contour_in_final,
            centroid_px=(float(cx), float(cy)),
            area_px=float(cv2.contourArea(contour_in_final)),
        )

        # Proxy A: centroid agreement bbox vs grid
        gi = grid_match[i]
        if gi is not None:
            gx, gy = blobs_grid[gi].centroid
            d_px = math.hypot(b.centroid[0] - gx, b.centroid[1] - gy)
            d_cells = d_px / max(cell_w, 1e-6)
            # Smooth falloff: 1 at d=0, ~0.5 at tol, ~0 well past tol.
            rec.bbox_grid_position = math.exp(
                -(d_cells / max(cons_cfg.position_tol_cells, 1e-6)) ** 2
            )
        else:
            rec.bbox_grid_position = 0.0

        # Proxy B: area ratio bbox vs grid
        if gi is not None:
            ga = blobs_grid[gi].area
            rec.bbox_grid_area = (
                min(b.area, ga) / max(b.area, ga) if max(b.area, ga) > 0
                else 0.0
            )
        else:
            rec.bbox_grid_area = 0.0

        # Proxy C: perturbation stability
        cx_list, cy_list, area_list = [b.centroid[0]], [b.centroid[1]], [b.area]
        for k, pl in enumerate(perturb_blobs):
            pi = perturb_matches[k][i]
            if pi is not None:
                cx_list.append(pl[pi].centroid[0])
                cy_list.append(pl[pi].centroid[1])
                area_list.append(pl[pi].area)
        if len(cx_list) >= 2:
            std_c = math.hypot(float(np.std(cx_list)), float(np.std(cy_list)))
            cov_a = float(np.std(area_list) / max(np.mean(area_list), 1e-6))
            # Normalize position std by cell width; combine.
            stability = (std_c / max(cell_w, 1e-6)) + cov_a
            rec.perturbation = math.exp(
                -(stability / max(cons_cfg.perturbation_stability_tol, 1e-6)) ** 2
            )
        else:
            rec.perturbation = 0.0

        # Combine (normalize weights so they always sum to 1)
        w = np.array([
            cons_cfg.w_bbox_grid_position,
            cons_cfg.w_bbox_grid_area,
            cons_cfg.w_perturbation,
        ], dtype=np.float64)
        w = w / max(w.sum(), 1e-9)
        rec.overall = float(
            w[0] * rec.bbox_grid_position
            + w[1] * rec.bbox_grid_area
            + w[2] * rec.perturbation
        )
        obstacles.append(rec)

    _emit("consistency", 1.0, f"Scored {len(obstacles)} obstacle(s)")
    return final_bbox, obstacles, stages_bbox
