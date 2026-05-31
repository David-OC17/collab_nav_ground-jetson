"""
transfer_obstacles.py
───────────────────────────────────────────────────────────────────────────────
Post-process a noisy/warped reconstructed drone map and project its detected
obstacles onto a clean background template.

Input
─────
  - reconstructed.png : warped aerial reconstruction. Color convention:
        * GREEN  (RGB 0,255,0)   = wall / arena boundary
        * PINK   (RGB 255,0,255) = obstacles (boxes, cones)
        * everything else        = noise / floor / artifacts
  - background.png    : clean, flat template (correct aspect, blue grid,
                        brown wall around) onto which obstacles are placed.

Pipeline
────────
  Stage 1 - Blue-grid de-warp
    Detect blue grid lines via HSV + HoughLinesP; compute axis directions;
    apply affine correction so vertical/horizontal grid lines become
    axis-aligned. Reused from map_to_occupancy.py.

  Stage 2 - Crop + mask-out everything outside the green wall
    a) Crop to the blue-grid boundary (rows/cols with >= N blue pixels).
    b) Build the green wall mask, take its convex hull (with dilation),
       zero out everything outside it. This kills the bright-pink noisy
       border that appears outside the real wall in the input.

  Stage 3 - Color masks + morphological closing
    Build clean pink (obstacle) and green (wall) masks. Apply MORPH_CLOSE
    `close_iterations` times to each (configurable).

  Stage 4 - Blob extraction
    Connected components on the cleaned pink mask. Blobs outside the
    configurable area band are dropped. Every surviving blob is accepted
    as an obstacle — no shape classification is applied (boxes may appear
    as joint/irregular shapes whose geometry doesn't fit a single-object
    heuristic).

  Stage 5 - Project obstacles onto background.png
    Two modes:
      * "bbox" : map the cleaned-image bounding box -> the inner wall bbox
                 of background.png, scale obstacle contours accordingly,
                 then draw filled.
      * "grid" : detect blue grid intersection lines in both images,
                 build a piecewise-linear cell-to-cell mapping for each
                 contour point. Better for non-uniform residual scaling.

Output
──────
  Final composited image with obstacles drawn (preserving their actual
  contour shape) on background.png, plus optional debug images for every
  pipeline stage.

Standalone use
──────────────
    python transfer_obstacles.py reconstructed.png \\
        --background background.png \\
        --out result.png \\
        --close-iters 3 \\
        --project-mode bbox \\
        --debug-dir debug/

Public API
──────────
    from transfer_obstacles import TransferConfig, run_pipeline
    final_bgr, stages = run_pipeline("reconstructed.png", "background.png", TransferConfig())
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


# ===============================================================================
# Configuration
# ===============================================================================

@dataclass
class ExpectedShape:
    """One expected obstacle class. `name` drives the heuristic branch
    ('box' / 'cone' have built-in heuristics; other names fall through to
    Florence-2 only). `descriptions` are free-text prompts used by
    Florence-2 verification."""
    name: str
    descriptions: List[str]
    draw_color_bgr: Tuple[int, int, int] = (255, 0, 255)  # default magenta


def _default_shapes() -> List[ExpectedShape]:
    return [
        ExpectedShape(
            name="box",
            descriptions=[
                "a cardboard box",
                "a rectangular box seen from above",
                "a square box",
            ],
            draw_color_bgr=(200, 120, 40),   # blue-ish (BGR)
        ),
        ExpectedShape(
            name="cone",
            descriptions=[
                "a traffic cone",
                "an orange safety cone",
                "a cone seen from above",
            ],
            draw_color_bgr=(0, 140, 255),    # orange (BGR)
        ),
    ]


@dataclass
class TransferConfig:
    # ── Stage 1: blue-grid de-warp ──────────────────────────────────────────
    correct_perspective: bool = True

    blue_h_lo: int = 95
    blue_h_hi: int = 135
    blue_s_lo: int = 60
    blue_s_hi: int = 255
    blue_v_lo: int = 60
    blue_v_hi: int = 255

    hough_threshold: int = 60
    hough_min_length: int = 40
    hough_max_gap: int = 20
    min_lines_per_axis: int = 3

    # ── Stage 2: crop ───────────────────────────────────────────────────────
    blue_edge_min_pixels: int = 8
    wall_hull_dilate_px: int = 8
    """Dilation applied to the green-wall convex hull before masking
    everything outside it to black. Larger -> more lenient (keeps a bit
    of pink/green that may sit just outside the hull)."""

    # ── Stage 3: pink/green color masks ─────────────────────────────────────
    # Pink/magenta is roughly H ~150-170 in OpenCV's H in [0,180].
    pink_h_lo: int = 140
    pink_h_hi: int = 170
    pink_s_lo: int = 120
    pink_s_hi: int = 255
    pink_v_lo: int = 80
    pink_v_hi: int = 255

    green_h_lo: int = 40
    green_h_hi: int = 85
    green_s_lo: int = 120
    green_s_hi: int = 255
    green_v_lo: int = 80
    green_v_hi: int = 255

    morph_kernel: int = 5
    close_iterations: int = 3
    """Number of MORPH_CLOSE passes applied to pink and green masks in
    Stage 3. Higher fills bigger gaps but may merge nearby blobs."""

    # ── Stage 4: blob filtering ─────────────────────────────────────────────
    min_blob_area_frac: float = 0.001
    """Drop blobs whose area / total-image-area is below this fraction.
    0.001 of a 2000x2000 image == 4000 px (a small but real object)."""
    max_blob_area_frac: float = 0.40
    """Drop blobs above this fraction (probably the wall or huge noise)."""

    min_blob_solidity: float = 0.25
    """Minimum solidity (area / convex-hull area) for a blob to be accepted.
    Rejects thin noise strips at the image border while passing any compact
    obstacle shape including irregular joint box arrangements (L, U, T, etc).
    Lower values are more permissive; 0.25 rejects only truly scattered noise."""

    expected_shapes: List[ExpectedShape] = field(default_factory=_default_shapes)

    # ── Stage 5: projection onto background ────────────────────────────────
    project_mode: str = "bbox"   # "bbox" or "grid"
    background_wall_h_lo: int = 5
    background_wall_h_hi: int = 25
    background_wall_s_lo: int = 50
    background_wall_v_lo: int = 40
    background_wall_v_hi: int = 200

    unknown_color_bgr: Tuple[int, int, int] = (200, 200, 200)
    """Color used when a blob doesn't match any expected shape (only
    drawn if Florence-2 is off and heuristic returns 'unknown' and
    drop_unknown=False)."""
    drop_unknown: bool = True
    """If True, discard blobs the heuristic can't classify as one of the
    expected_shapes. If False, draw them with unknown_color_bgr."""

    draw_outline_px: int = 2
    """Outline thickness around each drawn obstacle (0 = filled only)."""


# ===============================================================================
# Small helpers
# ===============================================================================

def _hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def _hsv_mask(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) -> np.ndarray:
    lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _blue_mask(img_bgr: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    return _hsv_mask(
        _hsv(img_bgr),
        cfg.blue_h_lo, cfg.blue_h_hi,
        cfg.blue_s_lo, cfg.blue_s_hi,
        cfg.blue_v_lo, cfg.blue_v_hi,
    )


def _pink_mask(img_bgr: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    return _hsv_mask(
        _hsv(img_bgr),
        cfg.pink_h_lo, cfg.pink_h_hi,
        cfg.pink_s_lo, cfg.pink_s_hi,
        cfg.pink_v_lo, cfg.pink_v_hi,
    )


def _green_mask(img_bgr: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    return _hsv_mask(
        _hsv(img_bgr),
        cfg.green_h_lo, cfg.green_h_hi,
        cfg.green_s_lo, cfg.green_s_hi,
        cfg.green_v_lo, cfg.green_v_hi,
    )


def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)


# ===============================================================================
# Stage 1 — Blue-grid de-warp  (ported from map_to_occupancy.py)
# ===============================================================================

def _detect_blue_lines(img: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    mask = _blue_mask(img, cfg)
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
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _split_hv(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.array([_line_angle_deg(*l) for l in lines])
    h_mask = (angles < 45) | (angles >= 135)
    return lines[h_mask], lines[~h_mask]


def _weighted_direction(lines: np.ndarray, force_positive: str = "x") -> Optional[np.ndarray]:
    if len(lines) == 0:
        return None

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.hypot(dx, dy) + 1e-9
    ux, uy = dx / lengths, dy / lengths

    flip = (uy < 0) if force_positive == "y" else (ux < 0)
    ux[flip] = -ux[flip]
    uy[flip] = -uy[flip]

    angles   = np.arctan2(uy, ux)
    weights  = lengths / lengths.sum()
    s_idx    = np.argsort(angles)
    cum_w    = np.cumsum(weights[s_idx])
    med_angle = angles[s_idx[np.searchsorted(cum_w, 0.5)]]
    return np.array([math.cos(med_angle), math.sin(med_angle)])


def dewarp(img: np.ndarray, cfg: TransferConfig, verbose: bool = True) -> np.ndarray:
    lines = _detect_blue_lines(img, cfg)

    if len(lines) < cfg.min_lines_per_axis * 2:
        _log(f"  [warn] Only {len(lines)} blue lines found — skipping de-warp.", verbose)
        return img

    h_lines, v_lines = _split_hv(lines)
    _log(f"  Blue lines: {len(lines)} total "
         f"({len(h_lines)} horizontal, {len(v_lines)} vertical)", verbose)

    d_h = _weighted_direction(h_lines, force_positive="x")
    d_v = _weighted_direction(v_lines, force_positive="y")

    if d_h is None and d_v is None:
        return img
    if d_h is None:
        d_h = np.array([d_v[1], -d_v[0]])
        if d_h[0] < 0:
            d_h = -d_h
    elif d_v is None:
        d_v = np.array([-d_h[1], d_h[0]])
        if d_v[1] < 0:
            d_v = -d_v

    A   = np.column_stack([d_h, d_v]).astype(np.float64)
    if float(np.linalg.det(A)) < 0:
        d_v = -d_v
        A   = np.column_stack([d_h, d_v]).astype(np.float64)

    cond = np.linalg.cond(A)
    if cond > 20:
        _log(f"  [warn] Axis nearly parallel (cond={cond:.1f}) — skipping.", verbose)
        return img

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return img

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
    _log(f"  De-warp applied -> {new_w}x{new_h} px (cond={cond:.2f})", verbose)
    return corrected


# ===============================================================================
# Stage 2 — Crop to blue boundary + zero-out outside green wall hull
# ===============================================================================

def _find_blue_boundary(img: np.ndarray, cfg: TransferConfig) -> Tuple[int, int, int, int]:
    mask = _blue_mask(img, cfg)
    h, w = mask.shape
    thr  = cfg.blue_edge_min_pixels

    row_counts = (mask > 0).sum(axis=1)
    col_counts = (mask > 0).sum(axis=0)

    top, bottom, left, right = 0, h - 1, 0, w - 1
    for r in range(h):
        if row_counts[r] >= thr:
            top = r
            break
    for r in range(h - 1, -1, -1):
        if row_counts[r] >= thr:
            bottom = r
            break
    for c in range(w):
        if col_counts[c] >= thr:
            left = c
            break
    for c in range(w - 1, -1, -1):
        if col_counts[c] >= thr:
            right = c
            break
    return top, bottom, left, right


def crop_to_blue(img: np.ndarray, cfg: TransferConfig, verbose: bool = True) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    top, bottom, left, right = _find_blue_boundary(img, cfg)
    _log(f"  Blue boundary: top={top} bottom={bottom} left={left} right={right} "
         f"(image {w_img}x{h_img})", verbose)

    if (bottom - top) < h_img * 0.10 or (right - left) < w_img * 0.10:
        _log("  [warn] Boundary too small — keeping full image.", verbose)
        return img
    return img[top:bottom + 1, left:right + 1].copy()


def mask_outside_wall(img: np.ndarray, cfg: TransferConfig,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a green-wall convex hull, dilate it slightly, and zero out
    everything outside it. Returns (cleaned_image, hull_mask).

    The hull is the arena's footprint as seen in the noisy reconstruction;
    the pink/green flecks outside it are stitching artifacts and must go.
    """
    g_mask = _green_mask(img, cfg)
    if g_mask.sum() == 0:
        _log("  [warn] No green wall pixels found — cannot mask outside.", verbose)
        return img.copy(), np.ones(img.shape[:2], dtype=np.uint8) * 255

    # Small open to drop speckles before finding contours / hull.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    g_clean = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN, k, iterations=1)
    if g_clean.sum() == 0:
        g_clean = g_mask

    pts = cv2.findNonZero(g_clean)
    hull = cv2.convexHull(pts)

    hull_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    if cfg.wall_hull_dilate_px > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.wall_hull_dilate_px * 2 + 1, cfg.wall_hull_dilate_px * 2 + 1),
        )
        hull_mask = cv2.dilate(hull_mask, dk, iterations=1)

    out = img.copy()
    out[hull_mask == 0] = (0, 0, 0)
    _log(f"  Outside-wall mask applied (hull area = {int(hull_mask.sum() / 255)} px)", verbose)
    return out, hull_mask


# ===============================================================================
# Stage 3 — Clean pink & green masks
# ===============================================================================

def build_clean_masks(img: np.ndarray, cfg: TransferConfig,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    pink  = _pink_mask(img,  cfg)
    green = _green_mask(img, cfg)

    if cfg.close_iterations > 0 and cfg.morph_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        pink  = cv2.morphologyEx(pink,  cv2.MORPH_CLOSE, k, iterations=cfg.close_iterations)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=cfg.close_iterations)
        # also a light open on pink to drop tiny specks
        pink  = cv2.morphologyEx(pink,  cv2.MORPH_OPEN,  k, iterations=1)

    _log(f"  Mask pixels: pink={int(pink.sum()/255)}  green={int(green.sum()/255)}  "
         f"(close iters = {cfg.close_iterations})", verbose)
    return pink, green


# ===============================================================================
# Stage 4 — Blob extraction
# ===============================================================================

@dataclass
class Blob:
    contour: np.ndarray          # Nx1x2 int32
    area: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    centroid: Tuple[float, float]
    final_label: Optional[str] = None

    # Consistency score in [0, 1] (1.0 == high confidence).
    # Populated by compute_consistency(); None means "not computed".
    consistency: Optional[float] = None
    consistency_components: Optional[Dict[str, float]] = None

    @property
    def area_frac(self) -> float:
        return self.area  # convenience placeholder; populated externally


def extract_blobs(pink_mask: np.ndarray, cfg: TransferConfig,
                  verbose: bool = True) -> List[Blob]:
    """Extract all pink blobs that fall within the area band.

    No shape classification is applied — every surviving blob is accepted as
    an obstacle (labeled with the first expected shape). Boxes may appear as
    joint or irregular arrangements whose geometry doesn't fit a per-object
    heuristic.
    """
    img_area = float(pink_mask.shape[0] * pink_mask.shape[1])
    min_area = cfg.min_blob_area_frac * img_area
    max_area = cfg.max_blob_area_frac * img_area

    label = cfg.expected_shapes[0].name if cfg.expected_shapes else "box"

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs: List[Blob] = []
    dropped_area = dropped_solidity = 0
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area or a > max_area:
            dropped_area += 1
            continue

        # Solidity filter: rejects thin noise strips (solidity ≈ 0.1-0.2)
        # while accepting any compact shape including joint box arrangements.
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) + 1e-9
        solidity = a / hull_area
        if solidity < cfg.min_blob_solidity:
            dropped_solidity += 1
            continue

        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        blobs.append(Blob(
            contour=c, area=a, bbox=(x, y, w, h), centroid=(cx, cy),
            final_label=label,
        ))

    _log(f"  Found {len(blobs)} blobs (dropped {dropped_area} by area, "
         f"{dropped_solidity} by solidity)  "
         f"area=[{min_area:.0f}..{max_area:.0f} px]  "
         f"min_solidity={cfg.min_blob_solidity}", verbose)
    for i, b in enumerate(blobs):
        _log(f"    [{i:02d}] area={b.area:7.0f}  centroid=({b.centroid[0]:.0f}, "
             f"{b.centroid[1]:.0f})  label={b.final_label}", verbose)
    return blobs



# ===============================================================================
# Stage 6 — Project onto background
# ===============================================================================

def _detect_background_wall_bbox(bg: np.ndarray,
                                 cfg: TransferConfig,
                                 verbose: bool = True
                                 ) -> Tuple[int, int, int, int]:
    """Return the inner bounding box (x, y, w, h) of the brown wall on
    background.png — i.e. the playable arena rectangle."""
    hsv = _hsv(bg)
    lo = np.array([cfg.background_wall_h_lo,
                   cfg.background_wall_s_lo,
                   cfg.background_wall_v_lo], dtype=np.uint8)
    hi = np.array([cfg.background_wall_h_hi,
                   255,
                   cfg.background_wall_v_hi], dtype=np.uint8)
    wall = cv2.inRange(hsv, lo, hi)
    pts = cv2.findNonZero(wall)
    if pts is None:
        _log("  [warn] No brown wall found in background — using full image bbox.",
             verbose)
        H, W = bg.shape[:2]
        return 0, 0, W, H
    x, y, w, h = cv2.boundingRect(pts)
    _log(f"  Background wall bbox: x={x} y={y} w={w} h={h}", verbose)
    return x, y, w, h


def _detect_grid_lines_xy(img: np.ndarray, cfg: TransferConfig
                          ) -> Tuple[List[int], List[int]]:
    """Return sorted unique x-coordinates of vertical blue lines and
    sorted y-coordinates of horizontal blue lines on the given image."""
    mask = _blue_mask(img, cfg)
    h, w = mask.shape

    col_counts = (mask > 0).sum(axis=0)
    row_counts = (mask > 0).sum(axis=1)

    col_thr = max(cfg.blue_edge_min_pixels, int(h * 0.30))
    row_thr = max(cfg.blue_edge_min_pixels, int(w * 0.30))

    def _peaks(counts: np.ndarray, thr: int) -> List[int]:
        peaks: List[int] = []
        in_peak = False
        start = 0
        for i, v in enumerate(counts):
            if v >= thr and not in_peak:
                in_peak = True
                start = i
            elif v < thr and in_peak:
                in_peak = False
                peaks.append((start + i - 1) // 2)
        if in_peak:
            peaks.append((start + len(counts) - 1) // 2)
        return peaks

    xs = _peaks(col_counts, col_thr)
    ys = _peaks(row_counts, row_thr)
    return xs, ys


def _interp_pos(value: float, src_grid: List[int], dst_grid: List[int]) -> float:
    """Piecewise-linear map of `value` from src grid coords to dst grid coords.
    Linearly extrapolates outside the grid using the nearest segment."""
    if not src_grid or not dst_grid or len(src_grid) != len(dst_grid):
        return value
    src = np.array(src_grid, dtype=np.float64)
    dst = np.array(dst_grid, dtype=np.float64)
    if value <= src[0]:
        if len(src) >= 2:
            t = (value - src[0]) / (src[1] - src[0] + 1e-9)
            return dst[0] + t * (dst[1] - dst[0])
        return dst[0]
    if value >= src[-1]:
        if len(src) >= 2:
            t = (value - src[-2]) / (src[-1] - src[-2] + 1e-9)
            return dst[-2] + t * (dst[-1] - dst[-2])
        return dst[-1]
    return float(np.interp(value, src, dst))


def project_onto_background(
    bg: np.ndarray,
    cleaned: np.ndarray,
    blobs: List[Blob],
    cfg: TransferConfig,
    verbose: bool = True,
) -> np.ndarray:
    """Composite obstacle contours onto bg. Returns a new image."""
    out = bg.copy()
    if not blobs:
        _log("  No blobs to project.", verbose)
        return out

    H_src, W_src = cleaned.shape[:2]
    bx, by, bw, bh = _detect_background_wall_bbox(bg, cfg, verbose)

    # Inner playable area = exclude the brown wall thickness. We estimate it
    # as ~3% of the bbox dimensions, but for the bbox-mode projection we map
    # the cleaned-image full extents onto bx..bx+bw, by..by+bh directly so
    # obstacles fall within the inner blue grid area.

    def _map_bbox(px: float, py: float) -> Tuple[int, int]:
        nx = px / max(W_src, 1)
        ny = py / max(H_src, 1)
        return int(round(bx + nx * bw)), int(round(by + ny * bh))

    map_grid_fn = None
    if cfg.project_mode == "grid":
        src_xs, src_ys = _detect_grid_lines_xy(cleaned, cfg)
        bg_xs,  bg_ys  = _detect_grid_lines_xy(bg, cfg)
        _log(f"  Grid lines: source ({len(src_xs)}x{len(src_ys)})  "
             f"background ({len(bg_xs)}x{len(bg_ys)})", verbose)

        if len(src_xs) >= 2 and len(bg_xs) >= 2 and len(src_ys) >= 2 and len(bg_ys) >= 2:
            # If counts differ, align by trimming the longer one symmetrically.
            def _align(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
                if len(a) == len(b):
                    return a, b
                if len(a) > len(b):
                    drop = len(a) - len(b)
                    left, right = drop // 2, drop - drop // 2
                    return a[left:len(a) - right], b
                drop = len(b) - len(a)
                left, right = drop // 2, drop - drop // 2
                return a, b[left:len(b) - right]
            xa, xb = _align(src_xs, bg_xs)
            ya, yb = _align(src_ys, bg_ys)

            def _map_grid(px: float, py: float) -> Tuple[int, int]:
                qx = _interp_pos(px, xa, xb)
                qy = _interp_pos(py, ya, yb)
                return int(round(qx)), int(round(qy))
            map_grid_fn = _map_grid
        else:
            _log("  [warn] grid mode requested but not enough lines on one side "
                 "— falling back to bbox.", verbose)

    map_fn = map_grid_fn if map_grid_fn is not None else _map_bbox

    # Build shape -> color lookup
    color_lookup = {s.name: s.draw_color_bgr for s in cfg.expected_shapes}

    drawn = 0
    for i, b in enumerate(blobs):
        label = b.final_label
        if label is None or label == "unknown":
            if cfg.drop_unknown:
                continue
            color = cfg.unknown_color_bgr
            label = "unknown"
        else:
            color = color_lookup.get(label, cfg.unknown_color_bgr)

        # Transform contour points
        pts_src = b.contour.reshape(-1, 2)
        pts_dst = np.array([map_fn(float(p[0]), float(p[1])) for p in pts_src],
                           dtype=np.int32).reshape(-1, 1, 2)

        cv2.drawContours(out, [pts_dst], -1, color, thickness=cv2.FILLED)
        if cfg.draw_outline_px > 0:
            # darker outline
            outline = tuple(int(c * 0.6) for c in color)
            cv2.drawContours(out, [pts_dst], -1, outline,
                             thickness=cfg.draw_outline_px)
        drawn += 1
        _log(f"    [{i:02d}] drew {label} blob ({len(pts_src)} pts)", verbose)

    _log(f"  Drew {drawn} obstacle(s) onto background.", verbose)
    return out


# ===============================================================================
# Debug image helpers
# ===============================================================================

def _save_debug(path: Optional[str], name: str, img: np.ndarray,
                debug_dir: Optional[str], verbose: bool = True):
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    p = os.path.join(debug_dir, f"{name}.png")
    cv2.imwrite(p, img)
    _log(f"    [debug] wrote {p}", verbose)


def _mask_to_bgr(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def _blob_overlay(img_bgr: np.ndarray, blobs: List[Blob]) -> np.ndarray:
    out = img_bgr.copy()
    for i, b in enumerate(blobs):
        if b.final_label == "box":
            color = (200, 120, 40)
        elif b.final_label == "cone":
            color = (0, 140, 255)
        else:
            color = (90, 90, 90)
        cv2.drawContours(out, [b.contour], -1, color, 2)
        cx, cy = int(b.centroid[0]), int(b.centroid[1])
        cv2.putText(out, f"#{i} {b.final_label or 'unknown'}", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ===============================================================================
# Consistency scoring (no ground truth)
# ===============================================================================
#
# We approximate per-blob detection confidence by running the pipeline N times
# with different parameters and projection modes, then matching blobs across
# runs. The proxies used:
#
#   A. bbox-vs-grid centroid agreement
#        Same blob, two projection modes → small distance in arena cells means
#        the geometry is internally consistent.
#
#   B. forward-project IoU against the source pink mask
#        Project the drawn blob from the background back into the cleaned-image
#        coordinate frame, intersect with the original pink mask. High overlap
#        = no information lost in the round trip.
#
#   C. parameter-perturbation stability (close_iters ± 1)
#        Run the heuristic three times with close_iters in {k-1, k, k+1} (or
#        clamped to >= 1). For matched blobs across runs, low std in centroid
#        and area means the detection is robust.
#
# Final score is a weighted average of A, B, C, each clipped to [0, 1].
# A blob present only in some runs (no match) gets a stability penalty.


@dataclass
class ConsistencyConfig:
    """How to compute the per-blob consistency score."""
    weight_agreement:   float = 0.40    # A
    weight_roundtrip:   float = 0.35    # B
    weight_stability:   float = 0.25    # C
    centroid_match_max_frac: float = 0.10
    """Two blobs from different runs are considered the same object if the
    distance between their centroids is < this fraction of the smaller image
    dimension."""
    enable_perturbation: bool = True
    """If False, skip the perturbation passes (C) and reweight A, B."""


def _match_blobs(ref: List[Blob], cand: List[Blob],
                 img_shape: Tuple[int, int],
                 max_frac: float) -> Dict[int, Optional[int]]:
    """Greedy nearest-centroid matching from `ref` to `cand`.
    Returns {ref_idx: cand_idx | None}."""
    if not ref or not cand:
        return {i: None for i in range(len(ref))}
    h, w = img_shape
    max_d = max_frac * min(h, w)

    out: Dict[int, Optional[int]] = {}
    used: set = set()
    for i, rb in enumerate(ref):
        best_j, best_d = None, float("inf")
        for j, cb in enumerate(cand):
            if j in used:
                continue
            dx = rb.centroid[0] - cb.centroid[0]
            dy = rb.centroid[1] - cb.centroid[1]
            d  = math.hypot(dx, dy)
            if d < best_d and d <= max_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
        out[i] = best_j
    return out


def _agreement_score(ref: Blob, alt: Optional[Blob],
                     img_shape: Tuple[int, int]) -> float:
    """Sub-component A: centroid + area agreement between two runs (e.g.
    bbox vs grid projection modes). Returns a value in [0, 1]."""
    if alt is None:
        return 0.0
    h, w = img_shape
    # Centroid distance normalized by image diagonal.
    diag = math.hypot(h, w) + 1e-9
    dx = ref.centroid[0] - alt.centroid[0]
    dy = ref.centroid[1] - alt.centroid[1]
    d_norm = math.hypot(dx, dy) / diag           # ~0 perfect; ~0.05 = 5% of diag
    pos_score = math.exp(-d_norm * 40.0)          # 0.5 at ~1.7% diag

    # Area ratio (always <= 1).
    a_ratio = min(ref.area, alt.area) / max(max(ref.area, alt.area), 1e-9)

    return 0.5 * pos_score + 0.5 * a_ratio


def _roundtrip_iou(blob: Blob, source_pink_mask: np.ndarray) -> float:
    """Sub-component B: IoU between the blob's contour (in cleaned-image
    coords) and the original pink mask after the closing operation."""
    if source_pink_mask is None:
        return 0.0
    h, w = source_pink_mask.shape[:2]
    drawn = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(drawn, [blob.contour], -1, 255, thickness=cv2.FILLED)
    inter = int(cv2.countNonZero(cv2.bitwise_and(drawn, source_pink_mask)))
    union = int(cv2.countNonZero(cv2.bitwise_or(drawn, source_pink_mask & drawn |
                                                drawn)))  # safer below
    # Compute union properly:
    union_mask = cv2.bitwise_or(drawn, source_pink_mask)
    # But we want IoU only over the local blob region; restricting union to
    # the dilated bbox keeps far-away pink from inflating the union.
    x, y, bw_, bh_ = blob.bbox
    pad = max(bw_, bh_) // 2
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw_ + pad); y1 = min(h, y + bh_ + pad)
    local_union = union_mask[y0:y1, x0:x1]
    local_inter = cv2.bitwise_and(drawn, source_pink_mask)[y0:y1, x0:x1]
    u = int(cv2.countNonZero(local_union)) + 1
    i = int(cv2.countNonZero(local_inter))
    return i / u


def _stability_score(matched_blobs: List[Optional[Blob]],
                     img_shape: Tuple[int, int]) -> float:
    """Sub-component C: low variance in centroid + area across N runs.
    Missing runs (None) are penalized linearly."""
    present = [b for b in matched_blobs if b is not None]
    if not present:
        return 0.0
    presence_ratio = len(present) / len(matched_blobs)

    cxs = np.array([b.centroid[0] for b in present])
    cys = np.array([b.centroid[1] for b in present])
    areas = np.array([b.area      for b in present])

    h, w = img_shape
    diag = math.hypot(h, w) + 1e-9
    # std of centroid as fraction of diagonal
    cstd = math.hypot(float(cxs.std()), float(cys.std())) / diag
    centroid_score = math.exp(-cstd * 50.0)

    if areas.mean() <= 0:
        area_score = 0.0
    else:
        cv_area = float(areas.std() / (areas.mean() + 1e-9))
        area_score = math.exp(-cv_area * 3.0)

    return presence_ratio * (0.5 * centroid_score + 0.5 * area_score)


def compute_consistency(
    primary_blobs:   List[Blob],
    primary_pink:    np.ndarray,
    primary_shape:   Tuple[int, int],
    alt_mode_blobs:  Optional[List[Blob]] = None,
    perturbed_runs:  Optional[List[List[Blob]]] = None,
    ccfg:            Optional[ConsistencyConfig] = None,
) -> None:
    """Populate `blob.consistency` and `blob.consistency_components` in place.

    Args:
        primary_blobs:   blobs from the primary run (canonical output).
        primary_pink:    cleaned pink mask from the primary run (for IoU).
        primary_shape:   (H, W) of the cleaned image (the coord frame the
                         primary contours live in).
        alt_mode_blobs:  blobs from a second run with the *other* projection
                         mode but same cleaned image (used for A).
        perturbed_runs:  list of blob-lists from perturbed close_iters runs
                         (used for C). Each list lives in the same coord frame
                         as primary_blobs because they share the dewarp+crop.
        ccfg:            scoring weights/thresholds.
    """
    ccfg = ccfg or ConsistencyConfig()
    H, W = primary_shape

    # Match the primary blobs to the alt-mode run (note: alt-mode has the SAME
    # contours in the cleaned-image frame; projection only happens later. So
    # 'agreement' here is really a robustness check, not a mode-difference
    # check, unless we explicitly run two separate primary pipelines. The ROS
    # node does that and passes the result here.)
    agree_match = _match_blobs(primary_blobs, alt_mode_blobs or [],
                               primary_shape, ccfg.centroid_match_max_frac)

    # Match the primary blobs to each perturbed run.
    perturbed_matches: List[List[Optional[Blob]]] = []
    if ccfg.enable_perturbation and perturbed_runs:
        for run in perturbed_runs:
            m = _match_blobs(primary_blobs, run, primary_shape,
                             ccfg.centroid_match_max_frac)
            perturbed_matches.append([run[j] if j is not None else None
                                      for j in m.values()])

    # Compute weights, dropping stability if no perturbed runs.
    wA = ccfg.weight_agreement
    wB = ccfg.weight_roundtrip
    wC = ccfg.weight_stability if perturbed_matches else 0.0
    s = wA + wB + wC
    if s <= 0:
        return
    wA, wB, wC = wA / s, wB / s, wC / s

    for i, b in enumerate(primary_blobs):
        # A: agreement with alt-mode run (e.g., the second projection mode
        # applied to the same blobs; here we use it as a generic second-run
        # cross-check).
        alt_b = (alt_mode_blobs[agree_match[i]]
                 if agree_match.get(i) is not None and alt_mode_blobs else None)
        a_score = _agreement_score(b, alt_b, primary_shape)

        # B: forward-project IoU vs source pink mask.
        b_score = _roundtrip_iou(b, primary_pink)

        # C: stability across perturbed runs (include self as run 0 so a blob
        # that's the only detection still scores nonzero presence).
        if perturbed_matches:
            stack: List[Optional[Blob]] = [b]
            for run in perturbed_matches:
                stack.append(run[i] if i < len(run) else None)
            c_score = _stability_score(stack, primary_shape)
        else:
            c_score = 0.0

        score = wA * a_score + wB * b_score + wC * c_score
        b.consistency = float(max(0.0, min(1.0, score)))
        b.consistency_components = {
            "agreement_A":  a_score,
            "roundtrip_B":  b_score,
            "stability_C":  c_score,
            "wA": wA, "wB": wB, "wC": wC,
        }


# ===============================================================================
# Public API
# ===============================================================================

def run_pipeline(
    reconstructed_path: str,
    background_path: str,
    cfg: Optional[TransferConfig] = None,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Full pipeline. Returns (final_image_bgr, stages_dict).

    stages_dict keys:
      "input", "dewarped", "cropped", "wall_masked",
      "pink_mask", "green_mask", "blob_overlay", "final"
    """
    cfg = cfg or TransferConfig()
    stages: Dict[str, np.ndarray] = {}
    sep = "-" * 60

    img = cv2.imread(reconstructed_path)
    if img is None:
        raise IOError(f"Cannot read reconstructed image: {reconstructed_path!r}")
    bg  = cv2.imread(background_path)
    if bg is None:
        raise IOError(f"Cannot read background image: {background_path!r}")
    stages["input"] = img.copy()

    _log(sep, verbose)
    _log(f"  Input:      {reconstructed_path}  [{img.shape[1]}x{img.shape[0]}]", verbose)
    _log(f"  Background: {background_path}  [{bg.shape[1]}x{bg.shape[0]}]", verbose)
    _log(sep, verbose)

    # Stage 1 ------------------------------------------------------------
    if cfg.correct_perspective:
        _log("\n[1/5] De-warping via blue grid lines...", verbose)
        dewarped = dewarp(img, cfg, verbose=verbose)
    else:
        _log("\n[1/5] Perspective correction disabled.", verbose)
        dewarped = img
    stages["dewarped"] = dewarped
    _save_debug(None, "01_dewarped", dewarped, debug_dir, verbose)

    # Stage 2 ------------------------------------------------------------
    _log("\n[2/5] Cropping to blue boundary...", verbose)
    cropped = crop_to_blue(dewarped, cfg, verbose=verbose)
    stages["cropped"] = cropped
    _save_debug(None, "02_cropped", cropped, debug_dir, verbose)

    _log("\n[2b/5] Masking everything outside the green wall hull...", verbose)
    cleaned, hull = mask_outside_wall(cropped, cfg, verbose=verbose)
    stages["wall_masked"] = cleaned
    _save_debug(None, "03_wall_masked", cleaned, debug_dir, verbose)

    # Stage 3 ------------------------------------------------------------
    _log("\n[3/5] Building pink & green masks with closing...", verbose)
    pink, green = build_clean_masks(cleaned, cfg, verbose=verbose)
    stages["pink_mask"]  = pink
    stages["green_mask"] = green
    _save_debug(None, "04_pink_mask",  _mask_to_bgr(pink,  (255,   0, 255)),
                debug_dir, verbose)
    _save_debug(None, "05_green_mask", _mask_to_bgr(green, (  0, 255,   0)),
                debug_dir, verbose)

    # Stage 4 ------------------------------------------------------------
    _log("\n[4/5] Extracting blobs...", verbose)
    blobs = extract_blobs(pink, cfg, verbose=verbose)

    overlay = _blob_overlay(cleaned, blobs)
    stages["blob_overlay"] = overlay
    _save_debug(None, "06_blob_overlay", overlay, debug_dir, verbose)

    # Stage 5 ------------------------------------------------------------
    _log(f"\n[5/5] Projecting blobs onto background "
         f"(mode={cfg.project_mode})...", verbose)
    final = project_onto_background(bg, cleaned, blobs, cfg, verbose=verbose)
    stages["final"] = final
    _save_debug(None, "07_final", final, debug_dir, verbose)

    _log(sep, verbose)
    return final, stages


# ===============================================================================
# CLI
# ===============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Clean a reconstructed drone map and transfer its "
                    "obstacles onto a flat background template."
    )
    ap.add_argument("image", help="Reconstructed drone map (PNG/JPG)")
    ap.add_argument("--background", "-b", required=True,
                    help="Background template image to draw onto")
    ap.add_argument("--out", "-o", default="result.png",
                    help="Path for the final composited image")
    ap.add_argument("--debug-dir",
                    help="If set, write per-stage debug images here")

    # Stage 1
    ap.add_argument("--no-dewarp", action="store_true",
                    help="Skip Stage 1 (blue-grid perspective correction)")

    # Stage 3
    ap.add_argument("--close-iters", type=int, default=3,
                    help="MORPH_CLOSE iterations on pink/green masks (default 3)")
    ap.add_argument("--morph-kernel", type=int, default=5,
                    help="Closing kernel size (default 5)")

    # Stage 4
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    ap.add_argument("--max-area-frac", type=float, default=0.40)
    ap.add_argument("--min-solidity", type=float, default=0.25,
                    help="Minimum blob solidity; filters noise strips (default 0.25)")

    # Stage 5
    ap.add_argument("--project-mode", default="bbox", choices=["bbox", "grid"],
                    help="bbox = simple scale-and-shift to background wall bbox; "
                         "grid = piecewise-linear via detected blue grid lines.")
    ap.add_argument("--keep-unknown", action="store_true",
                    help="Draw blobs the heuristic couldn't classify "
                         "(gray) instead of dropping them.")

    ap.add_argument("--quiet", action="store_true")
    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    cfg = TransferConfig(
        correct_perspective = not args.no_dewarp,
        close_iterations    = args.close_iters,
        morph_kernel        = args.morph_kernel,
        min_blob_area_frac  = args.min_area_frac,
        max_blob_area_frac  = args.max_area_frac,
        min_blob_solidity   = args.min_solidity,
        project_mode        = args.project_mode,
        drop_unknown        = not args.keep_unknown,
    )

    final, _stages = run_pipeline(
        args.image, args.background, cfg,
        debug_dir=args.debug_dir, verbose=not args.quiet,
    )
    cv2.imwrite(args.out, final)
    if not args.quiet:
        print(f"\nFinal image written -> {args.out}")


if __name__ == "__main__":
    main()
