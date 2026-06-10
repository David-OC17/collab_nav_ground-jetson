"""
map_diagnostics.py
─────────────────────────────────────────────────────────────────────────────
Raw diagnostic values from the stitching and transfer pipelines, collected
for downstream quality classification (e.g. XGBoost).

No pass/fail decisions are made here.  All outputs are measurements derived
post-hoc from:
  • the raw stitched map image  (np.ndarray from MapReconstructor.get_map())
  • the stages dict from run_pipeline() or compute_consistency()
  • (optional) List[ObstacleConsistency] from compute_consistency()
  • (optional) stitcher stats dict from MapReconstructor.stats
  • (optional) finalize report dict from MapReconstructor._last_finalize_report

Zero pipeline logic is modified or duplicated.  compute_diagnostics() is the
sole public entry point; it returns a MapDiagnostics dataclass that has a
to_feature_vector() helper producing a flat Dict[str, float] for XGBoost.

Typical usage
─────────────
    from map_diagnostics import DiagnosticsConfig, compute_diagnostics

    # After stitching and transfer:
    final_bgr, stages = run_pipeline(recon_path, bg_path, cfg)
    diag = compute_diagnostics(
        transfer_stages=stages,
        transfer_cfg=cfg,
        stitcher_stats=rec.stats,
        finalize_report=rec._last_finalize_report,
        stitched_map=rec_map_array,      # optional; falls back to stages["input"]
    )
    features = diag.to_feature_vector()  # flat dict, ready for XGBoost row

    # If the consistency pipeline was also run, pass its obstacles for richer
    # per-blob scores:
    final_bgr, obstacles, stages = compute_consistency(recon_path, bg_path, cfg)
    diag = compute_diagnostics(..., obstacles=obstacles)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .processing.transfer_obstacles import Blob, TransferConfig, build_clean_masks, extract_blobs
except ImportError:
    from arena_map_builder.processing.transfer_obstacles import (
        Blob, TransferConfig, build_clean_masks, extract_blobs)

# ObstacleConsistency is only used when the consistency pipeline was run;
# import it lazily so the module works without it.
_ObstacleConsistency = None


def _get_obstacle_consistency_class():
    global _ObstacleConsistency
    if _ObstacleConsistency is None:
        try:
            from .consistency import ObstacleConsistency as _OC
        except ImportError:
            try:
                from consistency import ObstacleConsistency as _OC
            except ImportError:
                _OC = None
        _ObstacleConsistency = _OC
    return _ObstacleConsistency


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticsConfig:
    """Tunable parameters for the diagnostic collectors."""

    arena_side_m: float = 3.90
    """Known arena side length (square). Used to express expected aspect ratio."""

    # ── Blue grid HSV (for line detection on the raw stitched map) ────────
    blue_h_lo: int = 90
    blue_h_hi: int = 130
    blue_s_lo: int = 50
    blue_v_lo: int = 50
    blue_hough_threshold: int = 50
    blue_hough_min_length: int = 40
    blue_hough_max_gap: int = 20

    # ── Green border HSV (mirroring TransferConfig defaults) ─────────────
    green_h_lo: int = 40
    green_h_hi: int = 85
    green_s_lo: int = 120
    green_v_lo: int = 80

    # ── Critical marker colors (BGR) ─────────────────────────────────────
    critical_marker_colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "amr":  (0,   0,   255),   # red
            "goal": (255, 255,   0),   # cyan
        }
    )
    """Mapping from marker role name to the BGR colour painted on the stitched
    map by the ArUco recolouring step. Keys become field prefixes in the
    feature vector (e.g. 'marker_amr_found', 'marker_goal_x_norm')."""

    marker_color_tol: int = 40
    """Per-channel BGR tolerance for marker detection (same as
    locate_color_marker_m default)."""

    marker_min_area_px: int = 9
    """Minimum contour area to accept a marker detection."""

    # ── Blob consistency threshold ────────────────────────────────────────
    blob_consistency_threshold: float = 0.35
    """Blobs with consistency >= this are counted as 'high-confidence'."""


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapDiagnostics:
    """
    All raw diagnostic values for one pipeline run.

    Fields are grouped by source:
      GROUP 1 — stitch_*      : spatial consistency of the raw stitched map
      GROUP 2 — dewarp_*      : dewarping quality (from transfer Stage 1)
      GROUP 3 — green_*       : green border hull metrics (from wall_masked)
      GROUP 4 — blob_*        : detected obstacle blobs
      GROUP 5 — marker_*      : critical ArUco marker localisation
      GROUP 6 — stitcher_* / pg_* : frame placement and pose-graph metrics

    None means the value could not be computed (missing input or absent
    feature).  to_feature_vector() maps None → -1.0 for XGBoost.
    """

    # ── GROUP 1: stitching spatial consistency ────────────────────────────
    stitch_h_line_count:     int    # horizontal blue lines found on raw stitched map
    stitch_v_line_count:     int    # vertical blue lines found
    stitch_h_angle_std_deg:  float  # std dev of H line angles (deg); 0 = perfect
    stitch_v_angle_std_deg:  float  # std dev of V line angles (deg)
    stitch_h_spacing_cv:     float  # CV of spacing between H lines; 0 = even grid
    stitch_v_spacing_cv:     float  # CV of spacing between V lines
    stitch_content_frac:     float  # non-black pixel fraction of stitched map
    stitch_convexity_ratio:  float  # non_black_area / convex_hull_area of content
    stitch_bbox_aspect_ratio: float # content bbox width / height (arena ≈ 1.0)

    # ── GROUP 2: dewarping quality ────────────────────────────────────────
    dewarp_h_line_count: int   # H lines found on the transfer input image
    dewarp_v_line_count: int   # V lines found
    dewarp_skipped:      bool  # True if dewarping was disabled or produced no change

    # ── GROUP 3: green border hull ────────────────────────────────────────
    green_hull_area_frac:  float  # hull_px / cleaned_image_px
    green_hull_convexity:  float  # green_px / hull_px; 1 = perfect ring, lower = scattered
    green_pixel_frac:      float  # raw green pixel count / image_px

    # ── GROUP 4: obstacles ────────────────────────────────────────────────
    blob_count:                   int            # blobs passing area + solidity filters
    blob_mean_area_frac:          float          # mean(blob.area / image_area)
    blob_std_area_frac:           float
    blob_max_area_frac:           float
    blob_mean_consistency:        Optional[float]  # None if consistency not computed
    blob_std_consistency:         Optional[float]
    blob_min_consistency:         Optional[float]
    blob_count_high_consistency:  Optional[int]    # blobs >= DiagnosticsConfig.threshold

    # ── GROUP 5: critical markers ─────────────────────────────────────────
    # Each dict is keyed by marker role name ("amr", "goal", …).
    marker_found:           Dict[str, bool]             # colour blob detected?
    marker_instance_count:  Dict[str, int]              # detections (1 = unique)
    marker_x_norm:          Dict[str, Optional[float]]  # [0,1] from image left
    marker_y_norm:          Dict[str, Optional[float]]  # [0,1] from image top
    marker_inside_hull:     Dict[str, Optional[bool]]   # inside green hull?
    inter_marker_distance_norm: Optional[float]         # px_dist / img_diag

    # ── GROUP 6: stitcher + pose-graph quality ────────────────────────────
    stitcher_frames_placed:      int
    stitcher_frames_failed:      int
    stitcher_placement_rate:     float          # placed / (placed + failed)
    stitcher_keyframes:          int
    stitcher_grid_refined_frac:  float          # grid_refined / placed (or 0)
    stitcher_grid_skipped_frac:  float          # grid_skipped / placed (or 0)
    pg_rms_before:    Optional[float]
    pg_rms_after:     Optional[float]
    pg_rms_ratio:     Optional[float]           # after / before; < 1 = improvement
    pg_marker_count:  Optional[int]
    pg_edge_count:    Optional[int]
    pg_frame_count:   Optional[int]

    # ─────────────────────────────────────────────────────────────────────

    def to_feature_vector(self) -> Dict[str, float]:
        """Flat scalar dict ready for XGBoost.

        • bool fields        → 0.0 / 1.0
        • None fields        → -1.0  (sentinel for 'not available')
        • Dict marker fields → expanded to marker_<role>_<field> keys
        """
        def _f(v: Any) -> float:
            if v is None:
                return -1.0
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            return float(v)

        out: Dict[str, float] = {
            # GROUP 1
            "stitch_h_line_count":      _f(self.stitch_h_line_count),
            "stitch_v_line_count":      _f(self.stitch_v_line_count),
            "stitch_h_angle_std_deg":   _f(self.stitch_h_angle_std_deg),
            "stitch_v_angle_std_deg":   _f(self.stitch_v_angle_std_deg),
            "stitch_h_spacing_cv":      _f(self.stitch_h_spacing_cv),
            "stitch_v_spacing_cv":      _f(self.stitch_v_spacing_cv),
            "stitch_content_frac":      _f(self.stitch_content_frac),
            "stitch_convexity_ratio":   _f(self.stitch_convexity_ratio),
            "stitch_bbox_aspect_ratio": _f(self.stitch_bbox_aspect_ratio),
            # GROUP 2
            "dewarp_h_line_count": _f(self.dewarp_h_line_count),
            "dewarp_v_line_count": _f(self.dewarp_v_line_count),
            "dewarp_skipped":      _f(self.dewarp_skipped),
            # GROUP 3
            "green_hull_area_frac": _f(self.green_hull_area_frac),
            "green_hull_convexity": _f(self.green_hull_convexity),
            "green_pixel_frac":     _f(self.green_pixel_frac),
            # GROUP 4
            "blob_count":                  _f(self.blob_count),
            "blob_mean_area_frac":         _f(self.blob_mean_area_frac),
            "blob_std_area_frac":          _f(self.blob_std_area_frac),
            "blob_max_area_frac":          _f(self.blob_max_area_frac),
            "blob_mean_consistency":       _f(self.blob_mean_consistency),
            "blob_std_consistency":        _f(self.blob_std_consistency),
            "blob_min_consistency":        _f(self.blob_min_consistency),
            "blob_count_high_consistency": _f(self.blob_count_high_consistency),
            # GROUP 5 — expand dicts
            **{f"marker_{role}_found":          _f(self.marker_found.get(role))
               for role in self.marker_found},
            **{f"marker_{role}_instance_count": _f(self.marker_instance_count.get(role))
               for role in self.marker_instance_count},
            **{f"marker_{role}_x_norm":         _f(self.marker_x_norm.get(role))
               for role in self.marker_x_norm},
            **{f"marker_{role}_y_norm":         _f(self.marker_y_norm.get(role))
               for role in self.marker_y_norm},
            **{f"marker_{role}_inside_hull":    _f(self.marker_inside_hull.get(role))
               for role in self.marker_inside_hull},
            "inter_marker_distance_norm": _f(self.inter_marker_distance_norm),
            # GROUP 6
            "stitcher_frames_placed":     _f(self.stitcher_frames_placed),
            "stitcher_frames_failed":     _f(self.stitcher_frames_failed),
            "stitcher_placement_rate":    _f(self.stitcher_placement_rate),
            "stitcher_keyframes":         _f(self.stitcher_keyframes),
            "stitcher_grid_refined_frac": _f(self.stitcher_grid_refined_frac),
            "stitcher_grid_skipped_frac": _f(self.stitcher_grid_skipped_frac),
            "pg_rms_before":    _f(self.pg_rms_before),
            "pg_rms_after":     _f(self.pg_rms_after),
            "pg_rms_ratio":     _f(self.pg_rms_ratio),
            "pg_marker_count":  _f(self.pg_marker_count),
            "pg_edge_count":    _f(self.pg_edge_count),
            "pg_frame_count":   _f(self.pg_frame_count),
        }
        return out

    def to_json(self) -> str:
        """Serialise to JSON (for saving per-run results to disk).

        Converts numpy types and Optional values to plain Python for
        json.dumps compatibility.
        """
        def _serial(v):
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, bool):
                return bool(v)
            if isinstance(v, dict):
                return {str(k): _serial(vv) for k, vv in v.items()}
            return v

        return json.dumps({
            k: _serial(getattr(self, k))
            for k in self.__dataclass_fields__
        }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blue_mask_local(img: np.ndarray, cfg: DiagnosticsConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg.blue_h_lo,   cfg.blue_s_lo, cfg.blue_v_lo],  dtype=np.uint8)
    hi = np.array([cfg.blue_h_hi,   255,            255],             dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _green_mask_local(img: np.ndarray, cfg: DiagnosticsConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg.green_h_lo, cfg.green_s_lo, cfg.green_v_lo], dtype=np.uint8)
    hi = np.array([cfg.green_h_hi, 255,             255],             dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _detect_grid_lines(
    img: np.ndarray,
    cfg: DiagnosticsConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (h_lines, v_lines) as (N, 4) arrays of [x1,y1,x2,y2].

    Replicates transfer_obstacles._detect_blue_lines + _split_hv without
    importing private functions.
    """
    mask = _blue_mask_local(img, cfg)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    raw = cv2.HoughLinesP(
        mask,
        rho=1, theta=np.pi / 180,
        threshold=cfg.blue_hough_threshold,
        minLineLength=cfg.blue_hough_min_length,
        maxLineGap=cfg.blue_hough_max_gap,
    )
    if raw is None:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, empty

    lines = raw.reshape(-1, 4).astype(np.float64)
    angles = np.degrees(
        np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
    ) % 180.0
    h_mask = (angles < 45) | (angles >= 135)
    return lines[h_mask], lines[~h_mask]


def _angle_std(lines: np.ndarray) -> float:
    """Std dev of line angles in degrees, wrapped to (−90, 90]."""
    if len(lines) < 2:
        return 0.0
    angles = np.degrees(
        np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
    )
    # Wrap to (−90, 90] so horizontal near-0° and near-180° don't inflate std.
    angles = ((angles + 90.0) % 180.0) - 90.0
    return float(np.std(angles))


def _spacing_cv(lines: np.ndarray, axis: int) -> float:
    """Coefficient of variation of inter-line spacing along `axis` (0=x, 1=y).

    Uses the midpoint of each line along the given axis to sort and compute
    gaps. Returns 0 when there are fewer than 2 lines (no gap to measure).
    """
    if len(lines) < 2:
        return 0.0
    # Midpoint along the axis of interest
    mids = (lines[:, axis] + lines[:, axis + 2]) / 2.0
    mids_sorted = np.sort(mids)
    gaps = np.diff(mids_sorted)
    gaps = gaps[gaps > 1]  # drop sub-pixel gaps (duplicate detections)
    if len(gaps) < 1:
        return 0.0
    mean = float(gaps.mean())
    if mean < 1e-6:
        return 0.0
    return float(gaps.std() / mean)


def _content_metrics(img: np.ndarray) -> Tuple[float, float, float]:
    """(content_frac, convexity_ratio, bbox_aspect_ratio) from non-black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    total_px = float(gray.shape[0] * gray.shape[1])
    nz = cv2.findNonZero((gray > 0).astype(np.uint8) * 255)
    if nz is None:
        return 0.0, 0.0, 1.0

    content_px = float(len(nz))
    content_frac = content_px / total_px

    hull = cv2.convexHull(nz)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    convexity = content_px / hull_area

    x, y, w, h = cv2.boundingRect(nz)
    aspect = float(w) / max(float(h), 1.0)
    return content_frac, convexity, aspect


def _green_hull_metrics(
    wall_masked: np.ndarray,
    cfg: DiagnosticsConfig,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """(hull_area_frac, hull_convexity, green_pixel_frac, hull_contour_or_None).

    hull_contour is returned so marker inside-hull checks can reuse it.
    """
    img_area = float(wall_masked.shape[0] * wall_masked.shape[1]) + 1e-6
    g = _green_mask_local(wall_masked, cfg)
    green_px = float(cv2.countNonZero(g))
    green_frac = green_px / img_area

    pts = cv2.findNonZero(g)
    if pts is None:
        return 0.0, 0.0, green_frac, None

    hull_cnt = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull_cnt)) + 1e-6
    hull_frac = hull_area / img_area
    convexity = green_px / hull_area
    return hull_frac, convexity, green_frac, hull_cnt


def _locate_marker_norm(
    img: np.ndarray,
    color_bgr: Tuple[int, int, int],
    tol: int,
    min_area_px: int,
) -> Tuple[int, Optional[float], Optional[float]]:
    """Locate a solid-colour marker blob in `img`.

    Returns (instance_count, x_norm, y_norm) where x_norm / y_norm are the
    centroid normalised to [0, 1] of the LARGEST blob.  x_norm is None when
    no blob is found.

    Replicates occupancy.locate_color_marker_m() but returns normalised
    pixel fractions instead of metres, so it has no dependency on
    OccupancyConfig.
    """
    b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
    lo = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)],   dtype=np.uint8)
    hi = np.array([min(255, b+tol), min(255, g+tol), min(255, r+tol)],   dtype=np.uint8)
    mask = cv2.inRange(img, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return 0, None, None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return len(contours), None, None

    H, W = img.shape[:2]
    cx_norm = (M["m10"] / M["m00"]) / max(W, 1)
    cy_norm = (M["m01"] / M["m00"]) / max(H, 1)
    return len(contours), float(cx_norm), float(cy_norm)


def _point_inside_hull(
    x_norm: float,
    y_norm: float,
    hull_cnt: np.ndarray,
    img_shape: Tuple[int, int],
) -> bool:
    """Return True if the normalised point (x_norm, y_norm) is inside hull_cnt."""
    H, W = img_shape
    px = float(x_norm * W)
    py = float(y_norm * H)
    return cv2.pointPolygonTest(hull_cnt, (px, py), measureDist=False) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — stitching spatial consistency
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stitch_metrics(
    stitched_map: np.ndarray,
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    h_lines, v_lines = _detect_grid_lines(stitched_map, cfg)
    content_frac, convexity, aspect = _content_metrics(stitched_map)
    return {
        "stitch_h_line_count":      len(h_lines),
        "stitch_v_line_count":      len(v_lines),
        "stitch_h_angle_std_deg":   _angle_std(h_lines),
        "stitch_v_angle_std_deg":   _angle_std(v_lines),
        # spacing: H lines spaced along Y axis, V lines along X axis
        "stitch_h_spacing_cv":      _spacing_cv(h_lines, axis=1),
        "stitch_v_spacing_cv":      _spacing_cv(v_lines, axis=0),
        "stitch_content_frac":      content_frac,
        "stitch_convexity_ratio":   convexity,
        "stitch_bbox_aspect_ratio": aspect,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — dewarping quality
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dewarp_metrics(
    stages: Dict[str, np.ndarray],
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    input_img    = stages.get("input")
    dewarped_img = stages.get("dewarped")

    if input_img is None:
        return {"dewarp_h_line_count": -1, "dewarp_v_line_count": -1,
                "dewarp_skipped": True}

    h_lines, v_lines = _detect_grid_lines(input_img, cfg)

    # If the dewarped image is the same object or same shape/content as the
    # input, the dewarp step was either disabled or found nothing to correct.
    skipped = (
        dewarped_img is None
        or dewarped_img is input_img
        or (dewarped_img.shape == input_img.shape
            and np.array_equal(dewarped_img, input_img))
    )
    return {
        "dewarp_h_line_count": len(h_lines),
        "dewarp_v_line_count": len(v_lines),
        "dewarp_skipped":      skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — green border hull
# ─────────────────────────────────────────────────────────────────────────────

def _compute_green_metrics(
    stages: Dict[str, np.ndarray],
    cfg: DiagnosticsConfig,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Returns (metrics_dict, hull_contour_or_None).

    The hull contour is threaded to GROUP 5 for inside-hull checks.
    """
    wall_masked = stages.get("wall_masked")
    if wall_masked is None:
        return {
            "green_hull_area_frac": -1.0,
            "green_hull_convexity": -1.0,
            "green_pixel_frac":     -1.0,
        }, None

    hull_frac, convexity, green_frac, hull_cnt = _green_hull_metrics(wall_masked, cfg)
    return {
        "green_hull_area_frac": hull_frac,
        "green_hull_convexity": convexity,
        "green_pixel_frac":     green_frac,
    }, hull_cnt


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4 — obstacles
# ─────────────────────────────────────────────────────────────────────────────

def _compute_blob_metrics(
    stages: Dict[str, np.ndarray],
    transfer_cfg: TransferConfig,
    diag_cfg: DiagnosticsConfig,
    obstacles: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Re-extracts blobs from stages["pink_mask"] (a pure function of that image,
    so it doesn't re-run any expensive processing).  If `obstacles` is a
    List[ObstacleConsistency] from the consistency pipeline, the per-obstacle
    overall scores are used for the consistency metrics.
    """
    pink = stages.get("pink_mask")
    if pink is None:
        return {
            "blob_count": 0,
            "blob_mean_area_frac": 0.0, "blob_std_area_frac": 0.0,
            "blob_max_area_frac":  0.0,
            "blob_mean_consistency": None, "blob_std_consistency": None,
            "blob_min_consistency":  None, "blob_count_high_consistency": None,
        }

    blobs: List[Blob] = extract_blobs(pink, transfer_cfg, verbose=False)
    img_area = float(pink.shape[0] * pink.shape[1]) + 1e-6
    areas = [b.area / img_area for b in blobs]

    out: Dict[str, Any] = {
        "blob_count":          len(blobs),
        "blob_mean_area_frac": float(np.mean(areas)) if areas else 0.0,
        "blob_std_area_frac":  float(np.std(areas))  if areas else 0.0,
        "blob_max_area_frac":  float(max(areas))      if areas else 0.0,
    }

    # Consistency scores — prefer ObstacleConsistency.overall if available,
    # fall back to Blob.consistency (set by transfer_obstacles.compute_consistency).
    OC = _get_obstacle_consistency_class()
    scores: Optional[List[float]] = None

    if obstacles is not None and OC is not None and len(obstacles) > 0:
        if isinstance(obstacles[0], OC):
            scores = [float(o.overall) for o in obstacles]

    if scores is None:
        # Try Blob.consistency (set when transfer_obstacles.compute_consistency
        # was called directly on the blobs from this run).
        blob_scores = [b.consistency for b in blobs if b.consistency is not None]
        if blob_scores:
            scores = [float(s) for s in blob_scores]

    if scores:
        thr = diag_cfg.blob_consistency_threshold
        out.update({
            "blob_mean_consistency":       float(np.mean(scores)),
            "blob_std_consistency":        float(np.std(scores)),
            "blob_min_consistency":        float(min(scores)),
            "blob_count_high_consistency": int(sum(s >= thr for s in scores)),
        })
    else:
        out.update({
            "blob_mean_consistency":       None,
            "blob_std_consistency":        None,
            "blob_min_consistency":        None,
            "blob_count_high_consistency": None,
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5 — critical markers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_marker_metrics(
    stages: Dict[str, np.ndarray],
    diag_cfg: DiagnosticsConfig,
    hull_cnt: Optional[np.ndarray],
) -> Dict[str, Any]:
    wall_masked = stages.get("wall_masked")
    colors = diag_cfg.critical_marker_colors

    found:          Dict[str, bool]             = {}
    instance_count: Dict[str, int]              = {}
    x_norm:         Dict[str, Optional[float]]  = {}
    y_norm:         Dict[str, Optional[float]]  = {}
    inside_hull:    Dict[str, Optional[bool]]   = {}

    for role, color in colors.items():
        if wall_masked is None:
            found[role]          = False
            instance_count[role] = 0
            x_norm[role]         = None
            y_norm[role]         = None
            inside_hull[role]    = None
            continue

        n, xn, yn = _locate_marker_norm(
            wall_masked, color,
            tol=diag_cfg.marker_color_tol,
            min_area_px=diag_cfg.marker_min_area_px,
        )
        found[role]          = n > 0
        instance_count[role] = n
        x_norm[role]         = xn
        y_norm[role]         = yn

        if xn is not None and hull_cnt is not None:
            inside_hull[role] = _point_inside_hull(
                xn, yn, hull_cnt, wall_masked.shape[:2]
            )
        else:
            inside_hull[role] = None

    # Inter-marker distance — only when exactly two markers are configured
    # and both are uniquely found.
    inter_dist: Optional[float] = None
    if wall_masked is not None and len(colors) == 2:
        roles = list(colors.keys())
        xn0, yn0 = x_norm.get(roles[0]), y_norm.get(roles[0])
        xn1, yn1 = x_norm.get(roles[1]), y_norm.get(roles[1])
        if xn0 is not None and yn0 is not None and xn1 is not None and yn1 is not None:
            H, W = wall_masked.shape[:2]
            diag_px = math.hypot(W, H)
            dx_px = (xn0 - xn1) * W
            dy_px = (yn0 - yn1) * H
            inter_dist = math.hypot(dx_px, dy_px) / max(diag_px, 1.0)

    return {
        "marker_found":           found,
        "marker_instance_count":  instance_count,
        "marker_x_norm":          x_norm,
        "marker_y_norm":          y_norm,
        "marker_inside_hull":     inside_hull,
        "inter_marker_distance_norm": inter_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 6 — stitcher + pose-graph quality
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stitcher_metrics(
    stitcher_stats: Optional[Dict],
    finalize_report: Optional[Dict],
) -> Dict[str, Any]:
    if stitcher_stats is None:
        stitcher_stats = {}

    placed  = int(stitcher_stats.get("placed",   0))
    failed  = int(stitcher_stats.get("failed",   0))
    total   = placed + failed
    rate    = float(placed) / max(total, 1)
    kfs     = int(stitcher_stats.get("keyframes", 0))
    refined = int(stitcher_stats.get("grid_refined", 0))
    skipped = int(stitcher_stats.get("grid_skipped", 0))

    out: Dict[str, Any] = {
        "stitcher_frames_placed":     placed,
        "stitcher_frames_failed":     failed,
        "stitcher_placement_rate":    rate,
        "stitcher_keyframes":         kfs,
        "stitcher_grid_refined_frac": float(refined) / max(placed, 1),
        "stitcher_grid_skipped_frac": float(skipped) / max(placed, 1),
    }

    if finalize_report is not None:
        rms_before = finalize_report.get("rms_before")
        rms_after  = finalize_report.get("rms_after")
        ratio = (
            float(rms_after) / max(float(rms_before), 1e-9)
            if rms_before is not None and rms_after is not None
            else None
        )
        out.update({
            "pg_rms_before":   float(rms_before) if rms_before is not None else None,
            "pg_rms_after":    float(rms_after)  if rms_after  is not None else None,
            "pg_rms_ratio":    ratio,
            "pg_marker_count": finalize_report.get("markers"),
            "pg_edge_count":   finalize_report.get("edges"),
            "pg_frame_count":  finalize_report.get("frames"),
        })
    else:
        out.update({
            "pg_rms_before": None, "pg_rms_after": None,
            "pg_rms_ratio":  None, "pg_marker_count": None,
            "pg_edge_count": None, "pg_frame_count":  None,
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(
    transfer_stages: Dict[str, np.ndarray],
    transfer_cfg: TransferConfig,
    stitched_map: Optional[np.ndarray] = None,
    stitcher_stats: Optional[Dict] = None,
    finalize_report: Optional[Dict] = None,
    obstacles: Optional[List] = None,
    cfg: Optional[DiagnosticsConfig] = None,
) -> MapDiagnostics:
    """Collect all diagnostic values for one pipeline run.

    Parameters
    ──────────
    transfer_stages : dict returned by run_pipeline() or compute_consistency().
        Must contain at least "input", "wall_masked", "pink_mask".
    transfer_cfg    : the TransferConfig used for that run (needed for blob
        re-extraction with the same area / solidity thresholds).
    stitched_map    : raw stitched BGR image (from MapReconstructor.get_map()).
        When None, stages["input"] is used as a fallback — the stitching
        metrics will then reflect the image AFTER it was loaded by the
        transfer pipeline (lossy round-trip through PNG), which is fine for
        relative comparisons but slightly less accurate.
    stitcher_stats  : dict from MapReconstructor.stats property.
    finalize_report : dict returned by MapReconstructor._last_finalize_report
        (the pose-graph solve report).
    obstacles       : optional List[ObstacleConsistency] from
        compute_consistency() for richer per-blob metrics.
    cfg             : DiagnosticsConfig; defaults are used when None.

    Returns
    ───────
    MapDiagnostics with all fields populated.  Call .to_feature_vector() to
    get a flat Dict[str, float] for XGBoost, or .to_json() to serialise to
    disk.
    """
    cfg = cfg or DiagnosticsConfig()

    # Use the stages["input"] image as fallback for stitching metrics.
    stitch_img = stitched_map if stitched_map is not None else transfer_stages.get("input")

    # ── collect all groups ──────────────────────────────────────────────
    g1 = _compute_stitch_metrics(stitch_img, cfg) if stitch_img is not None else {
        "stitch_h_line_count": 0, "stitch_v_line_count": 0,
        "stitch_h_angle_std_deg": 0.0, "stitch_v_angle_std_deg": 0.0,
        "stitch_h_spacing_cv": 0.0, "stitch_v_spacing_cv": 0.0,
        "stitch_content_frac": 0.0, "stitch_convexity_ratio": 0.0,
        "stitch_bbox_aspect_ratio": 1.0,
    }

    g2 = _compute_dewarp_metrics(transfer_stages, cfg)
    g3, hull_cnt = _compute_green_metrics(transfer_stages, cfg)
    g4 = _compute_blob_metrics(transfer_stages, transfer_cfg, cfg, obstacles)
    g5 = _compute_marker_metrics(transfer_stages, cfg, hull_cnt)
    g6 = _compute_stitcher_metrics(stitcher_stats, finalize_report)

    return MapDiagnostics(
        # GROUP 1
        stitch_h_line_count     = g1["stitch_h_line_count"],
        stitch_v_line_count     = g1["stitch_v_line_count"],
        stitch_h_angle_std_deg  = g1["stitch_h_angle_std_deg"],
        stitch_v_angle_std_deg  = g1["stitch_v_angle_std_deg"],
        stitch_h_spacing_cv     = g1["stitch_h_spacing_cv"],
        stitch_v_spacing_cv     = g1["stitch_v_spacing_cv"],
        stitch_content_frac     = g1["stitch_content_frac"],
        stitch_convexity_ratio  = g1["stitch_convexity_ratio"],
        stitch_bbox_aspect_ratio= g1["stitch_bbox_aspect_ratio"],
        # GROUP 2
        dewarp_h_line_count = g2["dewarp_h_line_count"],
        dewarp_v_line_count = g2["dewarp_v_line_count"],
        dewarp_skipped      = g2["dewarp_skipped"],
        # GROUP 3
        green_hull_area_frac = g3["green_hull_area_frac"],
        green_hull_convexity = g3["green_hull_convexity"],
        green_pixel_frac     = g3["green_pixel_frac"],
        # GROUP 4
        blob_count                  = g4["blob_count"],
        blob_mean_area_frac         = g4["blob_mean_area_frac"],
        blob_std_area_frac          = g4["blob_std_area_frac"],
        blob_max_area_frac          = g4["blob_max_area_frac"],
        blob_mean_consistency       = g4["blob_mean_consistency"],
        blob_std_consistency        = g4["blob_std_consistency"],
        blob_min_consistency        = g4["blob_min_consistency"],
        blob_count_high_consistency = g4["blob_count_high_consistency"],
        # GROUP 5
        marker_found               = g5["marker_found"],
        marker_instance_count      = g5["marker_instance_count"],
        marker_x_norm              = g5["marker_x_norm"],
        marker_y_norm              = g5["marker_y_norm"],
        marker_inside_hull         = g5["marker_inside_hull"],
        inter_marker_distance_norm = g5["inter_marker_distance_norm"],
        # GROUP 6
        stitcher_frames_placed     = g6["stitcher_frames_placed"],
        stitcher_frames_failed     = g6["stitcher_frames_failed"],
        stitcher_placement_rate    = g6["stitcher_placement_rate"],
        stitcher_keyframes         = g6["stitcher_keyframes"],
        stitcher_grid_refined_frac = g6["stitcher_grid_refined_frac"],
        stitcher_grid_skipped_frac = g6["stitcher_grid_skipped_frac"],
        pg_rms_before   = g6["pg_rms_before"],
        pg_rms_after    = g6["pg_rms_after"],
        pg_rms_ratio    = g6["pg_rms_ratio"],
        pg_marker_count = g6["pg_marker_count"],
        pg_edge_count   = g6["pg_edge_count"],
        pg_frame_count  = g6["pg_frame_count"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers — for inline use inside pipeline modules
# ─────────────────────────────────────────────────────────────────────────────

def compute_stitcher_diagnostics(
    stitched_map: np.ndarray,
    stitcher_stats: Optional[Dict] = None,
    finalize_report: Optional[Dict] = None,
    cfg: Optional[DiagnosticsConfig] = None,
) -> Dict[str, float]:
    """Compute Groups 1 + 6 from stitcher outputs only (no transfer pipeline).

    Called by MapReconstructor.finalize() while the canvas is still in memory
    so these features are captured without any extra pipeline work.

    Group 1 (stitch_*): grid line regularity, content convexity — derived
        from the stitched map image.
    Group 6 (stitcher_* / pg_*): frame placement stats, pose-graph residuals
        — derived from MapReconstructor.stats and _last_finalize_report.

    Returns a flat Dict[str, float] with the same key names as
    MapDiagnostics.to_feature_vector().  All None-valued fields use -1.0.
    """
    cfg = cfg or DiagnosticsConfig()
    g1 = _compute_stitch_metrics(stitched_map, cfg)
    g6 = _compute_stitcher_metrics(stitcher_stats, finalize_report)
    merged = {**g1, **g6}
    # Convert None → -1.0 and bool → 0.0/1.0 to match to_feature_vector()
    return {k: (-1.0 if v is None else (1.0 if v is True else (0.0 if v is False else float(v))))
            for k, v in merged.items()}


def compute_transfer_diagnostics(
    stages: Dict[str, np.ndarray],
    transfer_cfg: Any,
    cfg: Optional[DiagnosticsConfig] = None,
    obstacles: Optional[List] = None,
) -> Dict[str, float]:
    """Compute Groups 2-5 from transfer pipeline stages (no stitcher needed).

    Called by transfer_obstacles.run_pipeline() when return_diagnostics=True
    so the full transfer-side feature set is captured in one pipeline pass.

    Group 2 (dewarp_*):  blue-grid dewarping quality
    Group 3 (green_*):   arena border hull metrics
    Group 4 (blob_*):    detected obstacle blobs
    Group 5 (marker_*):  critical ArUco marker localisation

    Returns a flat Dict[str, float] with the same key names as
    MapDiagnostics.to_feature_vector().
    """
    cfg = cfg or DiagnosticsConfig()

    def _f(v: Any) -> float:
        if v is None:        return -1.0
        if v is True:        return  1.0
        if v is False:       return  0.0
        return float(v)

    g2 = _compute_dewarp_metrics(stages, cfg)
    g3, hull_cnt = _compute_green_metrics(stages, cfg)
    g4 = _compute_blob_metrics(stages, transfer_cfg, cfg, obstacles)
    g5 = _compute_marker_metrics(stages, cfg, hull_cnt)

    # Flatten g5's nested marker dicts — same expansion as to_feature_vector()
    flat_g5: Dict[str, float] = {}
    for role in g5.get("marker_found", {}):
        flat_g5[f"marker_{role}_found"]          = _f(g5["marker_found"].get(role))
        flat_g5[f"marker_{role}_instance_count"] = _f(g5["marker_instance_count"].get(role))
        flat_g5[f"marker_{role}_x_norm"]         = _f(g5["marker_x_norm"].get(role))
        flat_g5[f"marker_{role}_y_norm"]         = _f(g5["marker_y_norm"].get(role))
        flat_g5[f"marker_{role}_inside_hull"]    = _f(g5["marker_inside_hull"].get(role))
    flat_g5["inter_marker_distance_norm"] = _f(g5.get("inter_marker_distance_norm"))

    merged = {**g2, **g3, **g4, **flat_g5}
    return {k: _f(v) for k, v in merged.items()}