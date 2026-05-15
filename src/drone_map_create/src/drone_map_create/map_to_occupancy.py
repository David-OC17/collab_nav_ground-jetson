"""
map_to_occupancy.py
───────────────────────────────────────────────────────────────────────────────
Post-process a stitched drone-map image into a 2-D occupancy grid.

Pipeline
────────
  Stage 1 – Perspective / skew correction
    • Detect blue grid lines via HSV masking + HoughLinesP
    • Cluster into horizontal and vertical groups by angle
    • Compute length-weighted median direction for each axis independently
    • Apply a full affine correction (handles both rotation and keystone)

  Stage 2 – Map region extraction via blue grid-line boundary scan
    • Run AFTER perspective correction so the grid lines are axis-aligned
    • Starting from each image edge, scan row by row (top/bottom) and
      column by column (left/right)
    • The first row / column whose blue-pixel count >= blue_edge_min_pixels
      is treated as the map boundary for that side
    • Crop to those four limits so the outermost blue lines become flush
      with the image edge
    • Add a contour_thickness_px all-black border around the crop

  Stage 3 – Color segmentation -> occupancy values
    +--------------------------------------------------+
    | Color       | Label    | Grid value               |
    +--------------------------------------------------+
    | Black       | free     |  0  (arena floor)        |
    | Blue        | free     |  0  (grid lines)         |
    | White       | obstacle | 95  (pre-masked objects) |
    | Other       | occupied | 95  (treat as obstacle)  |
    +--------------------------------------------------+
    The added border ring is overridden to val_wall after segmentation.
    Obstacles are pre-masked to solid white by drone_map_gen.py before
    stitching; no colour-specific detection needed here.

  Stage 4 – Morphological closing on occupied pixels
    • Builds a binary mask of all occupied cells (value >= val_obstacle)
    • Applies MORPH_CLOSE with morph_kernel x morph_kernel ellipse,
      repeated morph_close_iterations times
    • Pixels newly covered by closing are written back as val_wall

  Stage 5 – Final value remapping
    • Remap to conventional encoding: free->0, occupied->100, unknown->-1
    • All cells with value in [0, 99]  -> intermediate_cell_value (default 25)
    • All cells with value == -1       -> intermediate_cell_value (default 25)
    • Final grid contains only: intermediate_cell_value and 100

Standalone use:
    python map_to_occupancy.py map.png --resolution 0.02 --save debug.png

Public API:
    from map_to_occupancy import MapConfig, process_map
    grid, debug_img = process_map("map.png", MapConfig())
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ===============================================================================
# Configuration
# ===============================================================================

@dataclass
class MapConfig:
    # -- resolution --------------------------------------------------------------
    resolution: float = 0.02
    """Metres per pixel in the final occupancy grid.  Tune based on your
    drone altitude and camera FOV.  Typical indoor arena: 0.01-0.05 m/px."""

    # -- pipeline control --------------------------------------------------------
    correct_perspective: bool = True
    """Set False to skip Stage 1 (blue-line affine de-warp) entirely.
    Useful when the drone was nearly vertical and residual skew is negligible,
    or while tuning colour thresholds in isolation."""

    # -- brown border (segmentation only - not used for crop boundary) -----------
    # Used only in Stage 3 to label brown pixels as walls.
    # The crop boundary is determined exclusively by blue grid lines (Stage 2).
    brown_h_lo: int = 5
    brown_h_hi: int = 25
    brown_s_lo: int = 50
    brown_s_hi: int = 255
    brown_v_lo: int = 40
    brown_v_hi: int = 180

    # -- blue grid-line detection -------------------------------------------------
    # Shared by Stage 1 (dewarp), Stage 2 (boundary scan), Stage 3 (free mask).
    blue_h_lo: int = 95
    blue_h_hi: int = 135
    blue_s_lo: int = 60
    blue_s_hi: int = 255
    blue_v_lo: int = 60
    blue_v_hi: int = 255

    hough_threshold: int = 60
    """Hough accumulator threshold.  Lower -> more (noisier) lines detected;
    higher -> fewer but stronger lines only."""
    hough_min_length: int = 40
    hough_max_gap: int = 20
    min_lines_per_axis: int = 3
    """Minimum blue lines required per axis to attempt de-rotation.
    If fewer are found the rotation step is skipped for that axis."""

    blue_edge_min_pixels: int = 8
    """Minimum blue-pixel count in a row or column for it to be recognised
    as a grid-line boundary when scanning inward from each image edge.
    Increase if noise near the edge causes a premature boundary hit;
    decrease if faint outermost grid lines are being skipped."""

    contour_thickness_px: int = 10
    """Thickness (px) of the all-black border added around the crop in
    Stage 2.  This border is overridden to val_wall in Stage 3 so it
    appears as fully occupied in the output."""

    # -- color segmentation ------------------------------------------------------
    # White obstacles: obstacles were pre-masked to solid white by drone_map_gen
    # before stitching, so detection is purely brightness + low saturation.
    white_s_hi: int = 40
    """Maximum saturation for a pixel to be classified as white obstacle.
    Pure white has S=0; raise slightly to catch off-white or slightly tinted
    surfaces under uneven lighting."""
    white_v_lo: int = 200
    """Minimum brightness (Value) for a pixel to be classified as white obstacle.
    Range [0,255].  Lower this if obstacles appear slightly grey in the map."""

    black_v_hi: int = 55
    """Pixels with HSV Value <= this are classified as the black floor (free)."""

    # -- morphology --------------------------------------------------------------
    morph_kernel: int = 5
    """Ellipse kernel size for per-mask open->close cleanup in Stage 3,
    and for the closing operation in Stage 4."""

    morph_close_iterations: int = 2
    """Number of times the Stage 4 closing operation is applied to the
    occupied mask.  Higher values bridge larger gaps between obstacle
    pixels and thicken thin walls.  Set to 0 to skip Stage 4 entirely."""

    # -- occupancy values --------------------------------------------------------
    val_free:     int = 0
    val_obstacle: int = 75   # kept for compatibility; not used in segmentation
    val_wall:     int = 95   # all occupied pixels (white obstacles + border)

    intermediate_cell_value: int = 25
    """Final remap target (Stage 5) for every cell whose value is either
    in [0, 99] (free / obstacle intermediate) or -1 (unknown).
    After Stage 5 the grid contains only this value and 100 (occupied).
    Set to 0 to disable remapping and keep strict 0 / 100 / -1."""


# ===============================================================================
# Stage 1 - Blue-line perspective / skew correction
# ===============================================================================

def _blue_mask(img_hsv: np.ndarray, cfg: MapConfig) -> np.ndarray:
    lo = np.array([cfg.blue_h_lo, cfg.blue_s_lo, cfg.blue_v_lo])
    hi = np.array([cfg.blue_h_hi, cfg.blue_s_hi, cfg.blue_v_hi])
    return cv2.inRange(img_hsv, lo, hi)


def _detect_blue_lines(img: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """Return Hough line segments (N x 4: x1,y1,x2,y2) from the blue mask."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _blue_mask(hsv, cfg)
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    lines = cv2.HoughLinesP(
        mask,
        rho=1, theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_length,
        maxLineGap=cfg.hough_max_gap,
    )
    return lines.reshape(-1, 4) if lines is not None else np.empty((0, 4))


def _line_angle_deg(x1, y1, x2, y2) -> float:
    """Angle in degrees, normalised to [0, 180)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _split_hv(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split into horizontal-ish (< 45 deg or >= 135 deg) and vertical-ish."""
    angles = np.array([_line_angle_deg(*l) for l in lines])
    h_mask = (angles < 45) | (angles >= 135)
    return lines[h_mask], lines[~h_mask]


def _weighted_direction(lines: np.ndarray, force_positive: str = "x") -> Optional[np.ndarray]:
    """
    Length-weighted median direction unit vector for a set of line segments.
    Returns a 2-vector (dx, dy), or None if lines is empty.
    force_positive: "x" for horizontal (rightward), "y" for vertical (downward).
    """
    if len(lines) == 0:
        return None

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.hypot(dx, dy) + 1e-9
    ux, uy = dx / lengths, dy / lengths

    flip = (uy < 0) if force_positive == "y" else (ux < 0)
    ux[flip] = -ux[flip]
    uy[flip] = -uy[flip]

    angles    = np.arctan2(uy, ux)
    weights   = lengths / lengths.sum()
    s_idx     = np.argsort(angles)
    cum_w     = np.cumsum(weights[s_idx])
    med_angle = angles[s_idx[np.searchsorted(cum_w, 0.5)]]
    return np.array([math.cos(med_angle), math.sin(med_angle)])


def dewarp(img: np.ndarray, cfg: MapConfig) -> Tuple[np.ndarray, dict]:
    """
    Use blue grid lines to compute and apply a full affine perspective correction.

    Detects horizontal and vertical blue line families, computes their actual
    (potentially skewed) axis directions, and builds the inverse affine that
    maps the image back to a true orthogonal grid.  Handles pure rotation AND
    keystone / shear simultaneously.

    Returns (corrected_image, info_dict).  Falls back to the original image
    if too few lines are detected or the axis directions are degenerate.
    """
    lines = _detect_blue_lines(img, cfg)
    info  = {"n_blue_lines": len(lines), "h_dir": None, "v_dir": None}

    if len(lines) < cfg.min_lines_per_axis * 2:
        print(f"  [warn] Only {len(lines)} blue lines found — skipping de-warp.")
        return img, info

    h_lines, v_lines = _split_hv(lines)
    print(f"  Blue lines: {len(lines)} total  "
          f"({len(h_lines)} horizontal, {len(v_lines)} vertical)")

    d_h = _weighted_direction(h_lines, force_positive="x")
    d_v = _weighted_direction(v_lines, force_positive="y")
    info.update({"h_dir": d_h, "v_dir": d_v})

    if d_h is None and d_v is None:
        return img, info

    # Derive missing axis from the other via 90-degree rotation
    if d_h is None:
        d_h = np.array([ d_v[1], -d_v[0]])
        if d_h[0] < 0:
            d_h = -d_h
    elif d_v is None:
        d_v = np.array([-d_h[1],  d_h[0]])
        if d_v[1] < 0:
            d_v = -d_v

    h_angle = math.degrees(math.atan2(d_h[1], d_h[0]))
    v_angle = math.degrees(math.atan2(d_v[1], d_v[0]))
    print(f"  H-axis: {h_angle:.2f}deg  |  V-axis: {v_angle:.2f}deg")

    A   = np.column_stack([d_h, d_v]).astype(np.float64)
    det = float(np.linalg.det(A))
    if det < 0:
        print("  [info] Left-handed basis (det<0) — flipping V direction.")
        d_v = -d_v
        A   = np.column_stack([d_h, d_v]).astype(np.float64)

    cond = np.linalg.cond(A)
    if cond > 20:
        print(f"  [warn] Axis directions nearly parallel (cond={cond:.1f}) "
              f"— skipping de-warp.")
        return img, info

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("  [warn] Degenerate axis directions — skipping de-warp.")
        return img, info

    ih, iw = img.shape[:2]
    cx, cy = iw / 2.0, ih / 2.0
    t      = np.array([cx, cy]) - A_inv @ np.array([cx, cy])
    M      = np.hstack([A_inv, t.reshape(2, 1)])

    corners_src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], dtype=np.float64)
    corners_dst = (A_inv @ corners_src.T).T + t
    x_min, y_min = corners_dst.min(axis=0)
    x_max, y_max = corners_dst.max(axis=0)
    new_w = int(math.ceil(x_max - x_min))
    new_h = int(math.ceil(y_max - y_min))

    M[0, 2] -= x_min
    M[1, 2] -= y_min

    corrected = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    print(f"  De-warp applied -> {new_w}x{new_h} px  (cond={cond:.2f})")
    return corrected, info


# ===============================================================================
# Stage 2 - Map region extraction via blue grid-line boundary scan
# ===============================================================================

def _find_blue_boundary(img: np.ndarray, cfg: MapConfig) -> Tuple[int, int, int, int]:
    """
    Scan inward from each image edge to find the first row / column that
    contains at least cfg.blue_edge_min_pixels blue pixels.

    This MUST be called on the PERSPECTIVE-CORRECTED image so the grid
    lines are axis-aligned and each line falls cleanly into a single row
    or column rather than cutting diagonally across many.

    Scan order:
      Top    -> rows 0, 1, 2, ... downward  until first qualifying row
      Bottom -> rows h-1, h-2, ...  upward  until first qualifying row
      Left   -> cols 0, 1, 2, ... rightward until first qualifying col
      Right  -> cols w-1, w-2, ... leftward until first qualifying col

    Returns (top, bottom, left, right) as inclusive pixel indices.
    Falls back to the image edge on any side where no blue is found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = _blue_mask(hsv, cfg)
    h, w = blue.shape
    thr  = cfg.blue_edge_min_pixels

    row_counts = (blue > 0).sum(axis=1)  # (h,) - blue pixels per row
    col_counts = (blue > 0).sum(axis=0)  # (w,) - blue pixels per col

    # Top: scan downward from row 0
    top = 0
    for r in range(h):
        if row_counts[r] >= thr:
            top = r
            break

    # Bottom: scan upward from last row
    bottom = h - 1
    for r in range(h - 1, -1, -1):
        if row_counts[r] >= thr:
            bottom = r
            break

    # Left: scan rightward from col 0
    left = 0
    for c in range(w):
        if col_counts[c] >= thr:
            left = c
            break

    # Right: scan leftward from last col
    right = w - 1
    for c in range(w - 1, -1, -1):
        if col_counts[c] >= thr:
            right = c
            break

    return top, bottom, left, right


def extract_map_region_by_blue(img: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """
    Crop the perspective-corrected image so that the outermost blue grid
    lines on each side become flush with the image border, then wrap the
    result in a solid black occupied border.

    Steps:
      1. Build blue HSV mask on the corrected image.
      2. Scan inward from each edge row-by-row / col-by-col.
      3. Crop to [top : bottom+1,  left : right+1].
      4. Add cfg.contour_thickness_px black border (overridden to val_wall
         in Stage 3 so it publishes as fully occupied).
    """
    h_img, w_img = img.shape[:2]
    top, bottom, left, right = _find_blue_boundary(img, cfg)

    print(
        f"  Blue boundary: top={top}  bottom={bottom}  "
        f"left={left}  right={right}  (image {w_img}x{h_img} px)"
    )

    if (bottom - top) < h_img * 0.10 or (right - left) < w_img * 0.10:
        print("  [warn] Blue boundary degenerate — returning full image.")
        return img

    cropped = img[top : bottom + 1, left : right + 1].copy()
    print(f"  Cropped to {cropped.shape[1]}x{cropped.shape[0]} px")

    brd = cfg.contour_thickness_px
    bordered = cv2.copyMakeBorder(
        cropped,
        brd, brd, brd, brd,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    print(f"  + {brd}px black border -> {bordered.shape[1]}x{bordered.shape[0]} px")
    return bordered


# ===============================================================================
# Stage 3 - Color segmentation -> occupancy
# ===============================================================================

def _morph_clean(mask: np.ndarray, ksize: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k)
    return m


def _hsv_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) -> np.ndarray:
    lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)

def _brown_mask(img_hsv: np.ndarray, cfg: MapConfig) -> np.ndarray:
    # Kept for reference; no longer called by the pipeline.
    lo = np.array([5,  50,  40])
    hi = np.array([25, 255, 180])
    return cv2.inRange(img_hsv, lo, hi)


def _white_mask(img_hsv: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """
    Detect solid-white obstacle pixels.

    In HSV space white has near-zero saturation and very high brightness.
    Obstacles are pre-masked to (255, 255, 255) by drone_map_gen.py before
    stitching, so this is essentially an exact-white detector with a small
    tolerance controlled by cfg.white_s_hi and cfg.white_v_lo.
    """
    lo = np.array([0,   0,              cfg.white_v_lo], dtype=np.uint8)
    hi = np.array([180, cfg.white_s_hi, 255           ], dtype=np.uint8)
    return cv2.inRange(img_hsv, lo, hi)


def segment(img: np.ndarray, cfg: MapConfig) -> Tuple[np.ndarray, dict]:
    """
    Segment img into per-pixel occupancy values.

    Three classes only:
      free     – black floor (low V), blue grid lines, or anything unclassified
      occupied – white pre-masked obstacles (high V, low S)

    Returns
    -------
    occ   : int8 array (H, W)
               0   free  (black floor + blue lines)
              95   occupied (white obstacle or unclassified)
    masks : dict label -> bool mask (for debug visualisation)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    k   = cfg.morph_kernel

    mask_blue = _morph_clean(
        _hsv_mask(hsv, cfg.blue_h_lo, cfg.blue_h_hi,
                       cfg.blue_s_lo, cfg.blue_s_hi,
                       cfg.blue_v_lo, cfg.blue_v_hi), k)

    mask_black = _morph_clean(
        cv2.inRange(hsv,
                    np.array([0,   0,   0             ], dtype=np.uint8),
                    np.array([180, 255, cfg.black_v_hi], dtype=np.uint8)), k)

    mask_white = _morph_clean(_white_mask(hsv, cfg), k)

    # Free: black floor OR blue grid lines OR anything unclassified
    mask_free = mask_black.astype(bool) | mask_blue.astype(bool)

    # Start everything as free, then paint occupied over white obstacles.
    occ = np.full(img.shape[:2], cfg.val_free, dtype=np.int8)
    occ[mask_white.astype(bool)] = cfg.val_wall

    masks = {
        "free":  mask_free,
        "blue":  mask_blue.astype(bool),
        "black": mask_black.astype(bool),
        "white": mask_white.astype(bool),
    }

    pcts = {k: f"{100*v.sum()/v.size:.1f}%" for k, v in masks.items()}
    print(f"  Segmentation coverage: {pcts}")
    return occ, masks


# ===============================================================================
# Stage 4 - Morphological closing on occupied pixels
# ===============================================================================

def morph_close_occupied(occ: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """
    Apply morphological closing to the binary occupied mask.

    Closing (dilation then erosion) bridges small gaps between nearby obstacle
    pixels and fills holes inside solid obstacles without significantly
    enlarging their footprint.

    Pixels that are not occupied before closing but are covered afterward are
    written back into occ as val_wall, preserving existing labels elsewhere.

    Controlled by cfg.morph_close_iterations.  Pass 0 to skip entirely.
    """
    if cfg.morph_close_iterations <= 0:
        print("  Skipped (morph_close_iterations=0).")
        return occ

    k        = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    occupied = (occ >= cfg.val_obstacle).astype(np.uint8) * 255
    closed   = cv2.morphologyEx(
        occupied, cv2.MORPH_CLOSE, k, iterations=cfg.morph_close_iterations
    )

    result  = occ.copy()
    new_occ = (closed > 0) & (occ < cfg.val_obstacle)
    result[new_occ] = cfg.val_wall

    print(f"  Closing ({cfg.morph_close_iterations} iter, "
          f"kernel={cfg.morph_kernel}px) added {int(new_occ.sum())} occupied pixels.")
    return result


# ===============================================================================
# Stage 5 - Build grid dict + final value remapping
# ===============================================================================

def make_occupancy_grid(occ: np.ndarray, cfg: MapConfig) -> dict:
    """
    Convert the int8 occupancy array into a plain Python dict.

    Two-pass remapping:
      Pass 1 - intermediate labels -> conventional encoding:
                 val_free           ->   0
                 >= val_wall (95)   -> 100
                 anything else      ->  -1  (should not occur after segmentation)

      Pass 2 - final remap (applied when intermediate_cell_value != 0):
                 -1           -> intermediate_cell_value  (safety catch)
                  0 .. 99    -> intermediate_cell_value
                  100         -> 100  (occupied; untouched)

    After Pass 2 the grid contains only two values:
      intermediate_cell_value  (free; navigable)
      100                      (occupied; obstacle / wall / border)

    Returned dict keys:
      "resolution"  float   - metres per pixel
      "width"       int     - columns
      "height"      int     - rows
      "data"        list    - flat int8 list, row-major, top row first
      "grid"        ndarray - 2-D int8 numpy array (height x width)
    """
    # Pass 1
    grid = np.where(occ == cfg.val_free, 0,
           np.where(occ >= cfg.val_obstacle, 100, -1)).astype(np.int8)

    # Pass 2
    tgt = cfg.intermediate_cell_value
    if tgt != 0:
        before = grid.copy()
        grid   = np.where(grid == -1,                   np.int8(tgt), grid)
        grid   = np.where((grid >= 0) & (grid <= 99),   np.int8(tgt), grid)
        n      = int((before != grid).sum())
        print(f"  Value remap: {n} cells -> {tgt}  "
              f"| occupied cells remaining: {int((grid == 100).sum())}")

    return {
        "resolution": cfg.resolution,
        "width":      occ.shape[1],
        "height":     occ.shape[0],
        "data":       grid.flatten().tolist(),
        "grid":       grid,
    }


# ===============================================================================
# Debug visualisation
# ===============================================================================

_PALETTE = {
    "free":  (180, 180,   0),   # dark cyan -- black floor + blue lines
    "blue":  (200, 100,   0),   # teal      -- blue grid lines (subset of free)
    "white": (  0,   0, 255),   # red       -- white pre-masked obstacles
}


def make_debug_image(
    img_orig: np.ndarray,
    occ: np.ndarray,
    masks: dict,
) -> np.ndarray:
    """
    Side-by-side debug image:
      Left  - original colour image
      Right - grayscale occupancy map with colour overlay per label
    """
    h, w = occ.shape

    occ_vis = np.where(occ == -1, 128,
              np.where(occ ==  0, 255,
              np.where(occ >= 75,   0, 200))).astype(np.uint8)
    occ_bgr = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)

    overlay = occ_bgr.copy()
    for label, color in _PALETTE.items():
        if label in masks and masks[label].any():
            overlay[masks[label]] = color

    orig_resized = cv2.resize(img_orig, (w, h))
    gap      = np.full((h, 10, 3), 200, dtype=np.uint8)
    combined = np.hstack([orig_resized, gap, overlay])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original",  (10,     30), font, 0.9, (0, 255, 0), 2)
    cv2.putText(combined, "Occupancy", (w + 20, 30), font, 0.9, (0, 255, 0), 2)

    legend_y = h - 160
    for i, (label, color) in enumerate(_PALETTE.items()):
        y = legend_y + i * 22
        cv2.rectangle(combined, (w + 20, y), (w + 40, y + 16), color, -1)
        cv2.putText(combined, label, (w + 46, y + 13), font, 0.55, (220, 220, 220), 1)

    return combined


# ===============================================================================
# Public API
# ===============================================================================

def process_map(
    image_path: str,
    cfg: Optional[MapConfig] = None,
    save_debug: Optional[str] = None,
    save_grid_png: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[dict, np.ndarray]:
    """
    Full pipeline: drone-map image -> (occupancy grid dict, debug image).

    Parameters
    ----------
    image_path    : path to the stitched map PNG/JPG
    cfg           : MapConfig (defaults used if None)
    save_debug    : if given, save the side-by-side debug image here
    save_grid_png : if given, save the grayscale occupancy PNG here
    verbose       : print progress

    Returns
    -------
    (grid, debug_img)

    grid keys:
      "resolution"  float   - metres per pixel
      "width"       int     - columns
      "height"      int     - rows
      "data"        list    - flat int8, row-major, top row first
      "grid"        ndarray - 2-D int8 array (height x width)
    """
    cfg = cfg or MapConfig()
    sep = "-" * 60

    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path!r}")
    if verbose:
        print(sep)
        print(f"  Input: {image_path}  [{img.shape[1]}x{img.shape[0]} px]")
        print(sep)

    # -- Stage 1: perspective / skew correction ---------------------------------
    if cfg.correct_perspective:
        if verbose:
            print("\n[1/5] De-warping via blue grid lines...")
        dewarped, _ = dewarp(img, cfg)
    else:
        if verbose:
            print("\n[1/5] Perspective correction disabled -- skipping.")
        dewarped = img

    # -- Stage 2: blue boundary scan + crop (on corrected image) ----------------
    if verbose:
        print("\n[2/5] Extracting map region (blue boundary scan)...")
    cropped = extract_map_region_by_blue(dewarped, cfg)

    # -- Stage 3: color segmentation --------------------------------------------
    if verbose:
        print("\n[3/5] Color segmentation...")
    occ, masks = segment(cropped, cfg)

    # Border stamp: the contour_thickness_px black border added in Stage 2
    # is caught as free (black floor) by segment(). Override it here so
    # it is correctly published as fully occupied wall.
    brd = cfg.contour_thickness_px
    if brd > 0:
        occ[:brd,  :]  = cfg.val_wall
        occ[-brd:, :]  = cfg.val_wall
        occ[:,  :brd]  = cfg.val_wall
        occ[:, -brd:]  = cfg.val_wall
        if verbose:
            print(f"  Border ring ({brd}px) stamped -> val_wall={cfg.val_wall}")

    # -- Stage 4: morphological closing on occupied pixels ----------------------
    if verbose:
        print(f"\n[4/5] Morphological closing "
              f"(iterations={cfg.morph_close_iterations})...")
    occ = morph_close_occupied(occ, cfg)

    # -- Stage 5: build grid + final value remap --------------------------------
    if verbose:
        print(f"\n[5/5] Building occupancy grid "
              f"(remap 0-99 and unknown -> {cfg.intermediate_cell_value})...")
    grid = make_occupancy_grid(occ, cfg)

    if verbose:
        real_w = grid["width"]  * cfg.resolution
        real_h = grid["height"] * cfg.resolution
        print(f"  Grid: {grid['width']}x{grid['height']} cells  "
              f"@ {cfg.resolution} m/px  ->  {real_w:.1f}x{real_h:.1f} m")

    # -- Debug images -----------------------------------------------------------
    debug = make_debug_image(cropped, occ, masks)

    if save_debug:
        cv2.imwrite(save_debug, debug)
        if verbose:
            print(f"\n  Debug image saved -> {save_debug}")

    if save_grid_png:
        g       = grid["grid"]
        tgt     = cfg.intermediate_cell_value
        occ_vis = np.where(g == 100, 0, np.where(g == tgt, 200, 128)).astype(np.uint8)
        cv2.imwrite(save_grid_png, occ_vis)
        if verbose:
            print(f"  Occupancy PNG saved -> {save_grid_png}")

    if verbose:
        print(sep)

    return grid, debug


# ===============================================================================
# CLI
# ===============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert a drone map image to a 2-D occupancy grid."
    )
    ap.add_argument("image", help="Path to the stitched map image (PNG/JPG)")
    ap.add_argument("--resolution",    type=float, default=0.02,
                    help="Metres per pixel (default: 0.02)")
    ap.add_argument("--save",          default="debug_occupancy.png",
                    help="Save side-by-side debug image here")
    ap.add_argument("--save-grid",     default="occupancy_grid.png",
                    help="Save grayscale occupancy PNG here")
    # Blue tuning
    ap.add_argument("--blue-h-lo",     type=int, default=95)
    ap.add_argument("--blue-h-hi",     type=int, default=135)
    ap.add_argument("--blue-edge-px",  type=int, default=8,
                    help="Min blue pixels per row/col to detect grid boundary")
    # White obstacle tuning
    ap.add_argument("--white-s-hi",    type=int, default=40,
                    help="Max saturation for white obstacle detection (default: 40)")
    ap.add_argument("--white-v-lo",    type=int, default=200,
                    help="Min brightness for white obstacle detection (default: 200)")
    # Black floor tuning
    ap.add_argument("--black-v-hi",    type=int, default=55)
    # Border / closing
    ap.add_argument("--contour-px",    type=int, default=10,
                    help="Thickness of the occupied border added after crop")
    ap.add_argument("--close-iters",   type=int, default=2,
                    help="Morphological closing iterations (0 = skip Stage 4)")
    # Final remap
    ap.add_argument("--intermediate",  type=int, default=25,
                    help="Value for all free/unknown cells in the final grid")
    args = ap.parse_args()

    cfg = MapConfig(
        resolution              = args.resolution,
        blue_h_lo               = args.blue_h_lo,
        blue_h_hi               = args.blue_h_hi,
        blue_edge_min_pixels    = args.blue_edge_px,
        white_s_hi              = args.white_s_hi,
        white_v_lo              = args.white_v_lo,
        black_v_hi              = args.black_v_hi,
        contour_thickness_px    = args.contour_px,
        morph_close_iterations  = args.close_iters,
        intermediate_cell_value = args.intermediate,
    )

    process_map(
        args.image, cfg,
        save_debug    = args.save,
        save_grid_png = args.save_grid,
    )