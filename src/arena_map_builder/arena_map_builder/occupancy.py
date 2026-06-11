"""
arena_map_builder.occupancy
─────────────────────────────────────────────────────────────────────────────
Convert the final composited map (background + drawn obstacles) into a
nav2-style OccupancyGrid.

Pixel-class semantics (initial scheme)
──────────────────────────────────────
  * Black floor / blue grid lines (inside arena)     →  10  (10% occupied)
  * Drawn obstacles (core)                            →  90  (90% occupied)
  * Brown wall around the arena                       → 100  (fully occupied)
  * White corner triangles outside the wall           →  -1  (unknown)

Uncertainty-aware thickening
────────────────────────────
For every obstacle we have a confidence c ∈ [0, 1] from the consistency
pass. We draw concentric rings around the core contour with thickness

    t = base_thickness_px * exp(-decay_rate * c)

so a low-confidence obstacle gets a larger uncertainty halo. Each ring is
drawn with a *lower* occupancy probability than the core, decaying
linearly from the core's 90% toward the background's 10% over the ring
count. The first pixel beyond the halo retains the background occupancy.

Coordinate convention
─────────────────────
Map origin at the bottom-left of the arena bbox (nav2 standard); +x
right, +y up. Internally the image is in top-left origin so the final
data array is flipped vertically before flattening into the
OccupancyGrid.data row-major buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np
import cv2

from .consistency import ObstacleConsistency


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class OccupancyConfig:
    # ── World sizing (REQUIRED for nav2 compatibility) ──────────────────
    resolution_m_per_cell: float = 0.05
    """Resolution in metres per OccupancyGrid cell."""
    arena_width_m:  float = 3.9
    arena_height_m: float = 3.9
    """Real-world arena dimensions (the bbox between the inner edges of
    the brown wall). The composited image's arena-bbox will be scaled to
    these dimensions before rasterization."""

    # ── Pixel-class occupancy values (0..100 or -1 for unknown) ─────────
    background_occ: int = 10    # black floor + blue grid lines
    obstacle_core_occ: int = 90 # solidly drawn obstacle
    wall_occ: int = 100         # brown perimeter wall
    unknown_occ: int = -1       # white corner triangles

    # ── Confidence-weighted thickening ──────────────────────────────────
    base_thickness_px: int = 8
    """Maximum halo thickness when confidence is 0."""
    decay_rate: float = 3.0
    """Exponential decay rate; halo_px = base * exp(-decay * c). At
    decay_rate=3.0 a confidence of 1.0 gives ~5% of the base thickness;
    confidence 0.5 gives ~22%."""
    halo_step_px: int = 2
    """Each concentric ring is `halo_step_px` thick. Smaller = smoother
    gradient but more dilations performed."""
    min_halo_occ_floor: Optional[int] = None
    """Halo rings linearly fade from obstacle_core_occ → this value over
    the halo thickness. None means use background_occ as the floor."""

    # ── Pixel-class detection thresholds (HSV) for the composited PNG ──
    # Brown wall (background.png) sits around HSV ~(16, 100, 150).
    # Orange-drawn cones sit at HSV ~(16, 255, 255). They share the hue
    # band, so we MUST separate by saturation — wall is desaturated,
    # cone is fully saturated. wall_s_hi must stay below ~180 to avoid
    # bleeding into cone color.
    wall_h_lo: int = 5
    wall_h_hi: int = 25
    wall_s_lo: int = 50
    wall_s_hi: int = 180
    wall_v_lo: int = 40
    wall_v_hi: int = 220

    white_v_lo: int = 230
    """V above which a pixel is considered the white outside-corner."""
    white_s_hi: int = 25
    """S below which (combined with white_v_lo) marks the unknown region."""

    # ── Misc ────────────────────────────────────────────────────────────
    frame_id: str = "world"


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _detect_arena_bbox(img_bgr: np.ndarray,
                       cfg: OccupancyConfig) -> Tuple[int, int, int, int]:
    """Inner bounding box of the brown wall = the playable arena."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([cfg.wall_h_lo, cfg.wall_s_lo, cfg.wall_v_lo], dtype=np.uint8)
    hi = np.array([cfg.wall_h_hi, cfg.wall_s_hi, cfg.wall_v_hi], dtype=np.uint8)
    wall = cv2.inRange(hsv, lo, hi)
    pts = cv2.findNonZero(wall)
    if pts is None:
        H, W = img_bgr.shape[:2]
        return 0, 0, W, H
    return cv2.boundingRect(pts)


def _build_class_masks(img_bgr: np.ndarray, cfg: OccupancyConfig
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wall_mask, unknown_mask) where each is uint8 {0, 255}."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    wall = cv2.inRange(
        hsv,
        np.array([cfg.wall_h_lo, cfg.wall_s_lo, cfg.wall_v_lo], dtype=np.uint8),
        np.array([cfg.wall_h_hi, cfg.wall_s_hi, cfg.wall_v_hi], dtype=np.uint8),
    )

    # Unknown = high V + low S (white background outside the rounded corners)
    h, s, v = cv2.split(hsv)
    unknown = ((v >= cfg.white_v_lo) & (s <= cfg.white_s_hi)).astype(np.uint8) * 255
    return wall, unknown


def _confidence_thickness(c: float, cfg: OccupancyConfig) -> int:
    """t = base * exp(-decay * c), clamped to >= 0."""
    c = float(np.clip(c, 0.0, 1.0))
    return max(0, int(round(cfg.base_thickness_px * math.exp(-cfg.decay_rate * c))))


# ───────────────────────────────────────────────────────────────────────────
# Main rasterizer
# ───────────────────────────────────────────────────────────────────────────

def rasterize_occupancy(
    final_bgr:   np.ndarray,
    obstacles:   List[ObstacleConsistency],
    cfg:         OccupancyConfig,
    debug_dir:   Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Build the int8 occupancy array (top-left-origin, suitable for
    later vertical-flip into nav2 OccupancyGrid.data).

    Returns
    -------
    occ_int8  : np.ndarray  (H, W) int8 in {-1, 0..100}
        Image-frame; the caller is responsible for the y-flip to nav2.
    arena_bbox: (x, y, w, h)
        The detected arena bbox in the input image's coords. Used by the
        caller to compute the OccupancyGrid origin in world coordinates.
    """
    H_img, W_img = final_bgr.shape[:2]

    # Start everything as background.
    occ = np.full((H_img, W_img), fill_value=cfg.background_occ, dtype=np.int16)

    # ── 1) Draw obstacles (with confidence-weighted thickening) FIRST.
    # Drawing them before walls means the wall/unknown overlay can be
    # restricted to "where it's still background", which protects
    # obstacles from being clobbered by stray brown-ish outline pixels.
    min_floor = (cfg.min_halo_occ_floor
                 if cfg.min_halo_occ_floor is not None
                 else cfg.background_occ)

    # Sort by ascending overall confidence so HIGH-confidence obstacles
    # are drawn LAST and their cores can never be obscured by a low-
    # confidence neighbour's halo.
    for obs in sorted(obstacles, key=lambda o: o.overall):
        c = float(np.clip(obs.overall, 0.0, 1.0))
        halo_px = _confidence_thickness(c, cfg)

        # Rasterize halo first, then core on top.
        if halo_px > 0 and cfg.halo_step_px > 0:
            n_rings = max(1, halo_px // cfg.halo_step_px)
            for r in range(n_rings, 0, -1):
                ring_px = r * cfg.halo_step_px
                ring_mask = np.zeros((H_img, W_img), dtype=np.uint8)
                cv2.drawContours(ring_mask, [obs.contour_px], -1,
                                 255, thickness=cv2.FILLED)
                if ring_px > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
                    )
                    ring_mask = cv2.dilate(ring_mask, k, iterations=1)
                # Linear interpolation: ring 1 (outermost) ≈ min_floor;
                # ring n_rings (innermost halo) ≈ core - one step.
                t = r / max(n_rings, 1)        # 1 == outermost
                ring_val = int(round(
                    cfg.obstacle_core_occ
                    + (min_floor - cfg.obstacle_core_occ) * t
                ))
                # Only upgrade (write higher occ). This is important
                # so neighbouring halos don't pull each other down.
                writeable = (ring_mask > 0) & (occ < ring_val)
                occ[writeable] = ring_val

        # Core (always full obstacle_core_occ)
        core_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        cv2.drawContours(core_mask, [obs.contour_px], -1,
                         255, thickness=cv2.FILLED)
        occ[core_mask > 0] = cfg.obstacle_core_occ

    # ── 2) Wall + unknown overlay — only where it's still background.
    # This protects obstacle pixels from being overwritten by accidental
    # brown-ish detections (e.g. an orange cone's outline).
    wall_mask, unknown_mask = _build_class_masks(final_bgr, cfg)
    background_cells = (occ == cfg.background_occ)
    occ[(wall_mask    > 0) & background_cells] = cfg.wall_occ
    occ[(unknown_mask > 0) & background_cells] = cfg.unknown_occ

    # Clip and downcast.
    occ_int8 = np.clip(occ, -1, 100).astype(np.int8)

    arena_bbox = _detect_arena_bbox(final_bgr, cfg)

    if debug_dir is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        # Visualize the occupancy grid as a grayscale image for inspection.
        vis = occ_int8.astype(np.int32)
        vis_img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        vis_img[vis == cfg.unknown_occ] = (128, 128, 128)   # grey = unknown
        known = vis != cfg.unknown_occ
        # Map 0..100 -> 0..255 brightness (darker = freer, brighter = occupied)
        v = np.clip(vis[known], 0, 100).astype(np.uint8)
        v = (v * (255 // 100)).astype(np.uint8)
        for ch in range(3):
            vis_img[..., ch][known] = v
        x, y, w, h = arena_bbox
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "08_occupancy_vis.png"), vis_img)

    return occ_int8, arena_bbox


# ───────────────────────────────────────────────────────────────────────────
# Marker localisation (goal / AMR fiducials masked as solid colour blocks)
# ───────────────────────────────────────────────────────────────────────────

def locate_color_marker_m(
    cleaned_bgr: np.ndarray,
    color_bgr:   Tuple[int, int, int],
    cfg:         OccupancyConfig,
    tol:         int = 40,
    min_area_px: int = 9,
) -> Optional[Tuple[float, float]]:
    """Locate a solid-colour marker block and return its centre in metres.

    The marker (e.g. an ArUco recoloured to a solid cyan/red block by the
    stitcher) is detected by colour in `cleaned_bgr` — the transfer pipeline's
    dewarped+cropped, wall-masked image (``stages["wall_masked"]``), which is
    the SAME frame the obstacles are projected from in bbox mode.

    Coordinate mapping (bbox projection)
    ────────────────────────────────────
    bbox-mode transfer maps the cleaned image's full extent linearly onto the
    background wall bbox, and occupancy_to_grid_array() crops that same wall
    bbox and scales it to (arena_width_m × arena_height_m). The intermediate
    bbox cancels, so a centroid (cx, cy) maps to:

        x_m = (cx / W_src) * arena_width_m
        y_m = (1 - cy / H_src) * arena_height_m     # occupancy origin = bottom-left

    which is exactly the transform an obstacle at the same pixel undergoes.

    Returns (x_m, y_m), or None when the colour is absent (marker not in map).
    """
    b, g, r = color_bgr
    lo = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8)
    hi = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8)
    mask = cv2.inRange(cleaned_bgr, lo, hi)

    # Light close to fuse interpolation-blurred edges into one solid blob.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    H_src, W_src = cleaned_bgr.shape[:2]
    x_m = (cx / max(W_src, 1)) * cfg.arena_width_m
    y_m = (1.0 - cy / max(H_src, 1)) * cfg.arena_height_m
    return (float(x_m), float(y_m))


# ───────────────────────────────────────────────────────────────────────────
# Final stage: build the OccupancyGrid msg (resampled to nav2 dims)
# ───────────────────────────────────────────────────────────────────────────

def occupancy_to_grid_array(
    occ_image_frame: np.ndarray,
    arena_bbox_px:   Tuple[int, int, int, int],
    cfg:             OccupancyConfig,
) -> Tuple[np.ndarray, int, int, Tuple[float, float]]:
    """
    Resample the per-pixel occupancy array to the world-resolution grid,
    then flip to nav2 (y-up) convention.

    Returns
    -------
    data         : flat int8 array of length width*height (nav2 row-major)
    width, height: OccupancyGrid dimensions in cells
    origin_xy_m  : world coordinates of the (0,0) cell (bottom-left of
                   the arena bbox)
    """
    # Number of cells across the real arena.
    width  = max(1, int(round(cfg.arena_width_m  / cfg.resolution_m_per_cell)))
    height = max(1, int(round(cfg.arena_height_m / cfg.resolution_m_per_cell)))

    x, y, w, h = arena_bbox_px
    if w <= 0 or h <= 0:
        # Degenerate; just return all-unknown.
        flat = np.full(width * height, cfg.unknown_occ, dtype=np.int8)
        return flat, width, height, (0.0, 0.0)

    cropped = occ_image_frame[y:y + h, x:x + w]

    # cv2.resize on int8 data: convert via int16 to avoid wrap-around.
    resized = cv2.resize(
        cropped.astype(np.int16),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int8)

    # Re-stamp the 2-cell wall border after resize. INTER_NEAREST can leave
    # partial or missing wall pixels at the very edge of the grid when the
    # arena bbox didn't land on an exact cell boundary. Overwriting the
    # perimeter guarantees a clean, gapless boundary in the published map.
    border = 2
    resized[:border,   :] = cfg.wall_occ
    resized[-border:,  :] = cfg.wall_occ
    resized[:,  :border] = cfg.wall_occ
    resized[:, -border:] = cfg.wall_occ

    # Flip vertically: image y grows downward, nav2 y grows upward.
    nav2_grid = np.flipud(resized)

    # Origin is the bottom-left corner of the arena bbox in world frame.
    # We treat (0, 0) world == bottom-left, so origin is (0, 0).
    origin_xy_m = (0.0, 0.0)

    return nav2_grid.flatten(), width, height, origin_xy_m
