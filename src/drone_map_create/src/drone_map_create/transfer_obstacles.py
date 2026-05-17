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

  Stage 4 - Blob extraction + heuristic shape classification
    Connected components on the cleaned pink mask. For each blob:
      - compute solidity, aspect ratio, vertex count of approxPolyDP, etc.
      - decide whether it looks like a "box" (rectangular), a "cone"
        (round or triangular), or "unknown".
    Tiny / scattered blobs are dropped via area / solidity thresholds.

  Stage 5 - (Optional) Florence-2 verification
    If enabled, every accepted blob is crop-padded and sent to Florence-2
    with the shape descriptions. Blobs that don't match any expected shape
    are rejected. Soft import — clear error if transformers is unavailable.

  Stage 6 - Project obstacles onto background.png
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
            draw_color_bgr=(200, 120, 40),  # blue-ish (BGR)
        ),
        ExpectedShape(
            name="cone",
            descriptions=[
                "a traffic cone",
                "an orange safety cone",
                "a cone seen from above",
            ],
            draw_color_bgr=(0, 140, 255),  # orange (BGR)
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

    # ── Stage 4: blob filtering + heuristic classification ──────────────────
    min_blob_area_frac: float = 0.001
    """Drop blobs whose area / total-image-area is below this fraction.
    0.001 of a 2000x2000 image == 4000 px (a small but real object)."""
    max_blob_area_frac: float = 0.25
    """Drop blobs above this fraction (probably the wall or huge noise)."""

    min_solidity_box: float = 0.85
    """Solidity (area / convex-hull area) above which a blob looks like
    a filled, convex shape (consistent with a box or upright cone)."""
    box_aspect_min: float = 0.55
    box_aspect_max: float = 1.85
    """Aspect-ratio band for box classification on the min-area rect."""
    box_rectangularity_min: float = 0.78
    """Area / minAreaRect area for box classification."""

    cone_circularity_min: float = 0.55
    """4*pi*A / P^2 lower bound for a cone seen from above (circle)."""
    cone_triangle_vertices: Tuple[int, int] = (3, 5)
    """approxPolyDP vertex range for a triangle-like cone."""

    # ── Stage 5: Florence-2 verification (opt-in) ──────────────────────────
    use_florence2: bool = False
    florence2_model_id: str = "microsoft/Florence-2-base"
    florence2_device: str = "auto"  # "auto" / "cpu" / "cuda"
    florence2_pad_px: int = 20
    """Pixels of context padding around the blob crop sent to Florence-2."""
    florence2_min_score: float = 0.0
    """Florence-2 doesn't always return scores; placeholder for future use."""

    expected_shapes: List[ExpectedShape] = field(default_factory=_default_shapes)

    # ── Stage 6: projection onto background ────────────────────────────────
    project_mode: str = "bbox"  # "bbox" or "grid"
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
        cfg.blue_h_lo,
        cfg.blue_h_hi,
        cfg.blue_s_lo,
        cfg.blue_s_hi,
        cfg.blue_v_lo,
        cfg.blue_v_hi,
    )


def _pink_mask(img_bgr: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    return _hsv_mask(
        _hsv(img_bgr),
        cfg.pink_h_lo,
        cfg.pink_h_hi,
        cfg.pink_s_lo,
        cfg.pink_s_hi,
        cfg.pink_v_lo,
        cfg.pink_v_hi,
    )


def _green_mask(img_bgr: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    return _hsv_mask(
        _hsv(img_bgr),
        cfg.green_h_lo,
        cfg.green_h_hi,
        cfg.green_s_lo,
        cfg.green_s_hi,
        cfg.green_v_lo,
        cfg.green_v_hi,
    )


def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)


# ===============================================================================
# Stage 1 — Blue-grid de-warp  (ported from map_to_occupancy.py)
# ===============================================================================


def _detect_blue_lines(img: np.ndarray, cfg: TransferConfig) -> np.ndarray:
    mask = _blue_mask(img, cfg)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
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


def _weighted_direction(
    lines: np.ndarray, force_positive: str = "x"
) -> Optional[np.ndarray]:
    if len(lines) == 0:
        return None

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.hypot(dx, dy) + 1e-9
    ux, uy = dx / lengths, dy / lengths

    flip = (uy < 0) if force_positive == "y" else (ux < 0)
    ux[flip] = -ux[flip]
    uy[flip] = -uy[flip]

    angles = np.arctan2(uy, ux)
    weights = lengths / lengths.sum()
    s_idx = np.argsort(angles)
    cum_w = np.cumsum(weights[s_idx])
    med_angle = angles[s_idx[np.searchsorted(cum_w, 0.5)]]
    return np.array([math.cos(med_angle), math.sin(med_angle)])


def dewarp(img: np.ndarray, cfg: TransferConfig, verbose: bool = True) -> np.ndarray:
    lines = _detect_blue_lines(img, cfg)

    if len(lines) < cfg.min_lines_per_axis * 2:
        _log(
            f"  [warn] Only {len(lines)} blue lines found — skipping de-warp.", verbose
        )
        return img

    h_lines, v_lines = _split_hv(lines)
    _log(
        f"  Blue lines: {len(lines)} total "
        f"({len(h_lines)} horizontal, {len(v_lines)} vertical)",
        verbose,
    )

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

    A = np.column_stack([d_h, d_v]).astype(np.float64)
    if float(np.linalg.det(A)) < 0:
        d_v = -d_v
        A = np.column_stack([d_h, d_v]).astype(np.float64)

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
    t = np.array([cx, cy]) - A_inv @ np.array([cx, cy])
    M = np.hstack([A_inv, t.reshape(2, 1)])

    corners_src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], dtype=np.float64)
    corners_dst = (A_inv @ corners_src.T).T + t
    x_min, y_min = corners_dst.min(axis=0)
    x_max, y_max = corners_dst.max(axis=0)
    new_w = int(math.ceil(x_max - x_min))
    new_h = int(math.ceil(y_max - y_min))

    M[0, 2] -= x_min
    M[1, 2] -= y_min

    corrected = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    _log(f"  De-warp applied -> {new_w}x{new_h} px (cond={cond:.2f})", verbose)
    return corrected


# ===============================================================================
# Stage 2 — Crop to blue boundary + zero-out outside green wall hull
# ===============================================================================


def _find_blue_boundary(
    img: np.ndarray, cfg: TransferConfig
) -> Tuple[int, int, int, int]:
    mask = _blue_mask(img, cfg)
    h, w = mask.shape
    thr = cfg.blue_edge_min_pixels

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


def crop_to_blue(
    img: np.ndarray, cfg: TransferConfig, verbose: bool = True
) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    top, bottom, left, right = _find_blue_boundary(img, cfg)
    _log(
        f"  Blue boundary: top={top} bottom={bottom} left={left} right={right} "
        f"(image {w_img}x{h_img})",
        verbose,
    )

    if (bottom - top) < h_img * 0.10 or (right - left) < w_img * 0.10:
        _log("  [warn] Boundary too small — keeping full image.", verbose)
        return img
    return img[top : bottom + 1, left : right + 1].copy()


def mask_outside_wall(
    img: np.ndarray, cfg: TransferConfig, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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
    _log(
        f"  Outside-wall mask applied (hull area = {int(hull_mask.sum() / 255)} px)",
        verbose,
    )
    return out, hull_mask


# ===============================================================================
# Stage 3 — Clean pink & green masks
# ===============================================================================


def build_clean_masks(
    img: np.ndarray, cfg: TransferConfig, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    pink = _pink_mask(img, cfg)
    green = _green_mask(img, cfg)

    if cfg.close_iterations > 0 and cfg.morph_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        pink = cv2.morphologyEx(
            pink, cv2.MORPH_CLOSE, k, iterations=cfg.close_iterations
        )
        green = cv2.morphologyEx(
            green, cv2.MORPH_CLOSE, k, iterations=cfg.close_iterations
        )
        # also a light open on pink to drop tiny specks
        pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, k, iterations=1)

    _log(
        f"  Mask pixels: pink={int(pink.sum()/255)}  green={int(green.sum()/255)}  "
        f"(close iters = {cfg.close_iterations})",
        verbose,
    )
    return pink, green


# ===============================================================================
# Stage 4 — Blob extraction + heuristic shape classification
# ===============================================================================


@dataclass
class Blob:
    contour: np.ndarray  # Nx1x2 int32
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float]
    solidity: float
    aspect: float
    rectangularity: float
    circularity: float
    n_vertices: int
    heuristic_label: str  # "box" / "cone" / "unknown"
    heuristic_confidence: float  # 0..1
    final_label: Optional[str] = None  # set after classify / Florence-2

    @property
    def area_frac(self) -> float:
        return self.area  # convenience placeholder; populated externally


def _classify_blob(
    contour: np.ndarray, cfg: TransferConfig
) -> Tuple[str, float, Dict[str, float]]:
    """
    Returns (label, confidence, metrics_dict).
    Heuristic only — lenient: prefers to label something rather than 'unknown'
    when borderline. Florence-2 can override later if enabled.
    """
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return "unknown", 0.0, {}

    peri = cv2.arcLength(contour, closed=True) + 1e-9
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = area / hull_area

    rect = cv2.minAreaRect(contour)  # ((cx,cy),(w,h),angle)
    w, h = rect[1]
    rect_a = max(w * h, 1e-9)
    rectangularity = area / rect_a
    if w == 0 or h == 0:
        aspect = 0.0
    else:
        aspect = min(w, h) / max(w, h)  # 1.0 == square, <1 == elongated

    circularity = 4.0 * math.pi * area / (peri * peri)

    approx = cv2.approxPolyDP(contour, 0.04 * peri, closed=True)
    n_vert = len(approx)

    metrics = dict(
        area=area,
        solidity=solidity,
        aspect=aspect,
        rectangularity=rectangularity,
        circularity=circularity,
        n_vertices=float(n_vert),
    )

    # Box test: high rectangularity, decent solidity, moderate aspect.
    is_box = (
        rectangularity >= cfg.box_rectangularity_min
        and solidity >= cfg.min_solidity_box
        and cfg.box_aspect_min <= aspect <= cfg.box_aspect_max
    )

    # Cone-circle: high circularity + decent solidity.
    is_cone_circle = (
        circularity >= cfg.cone_circularity_min
        and solidity >= cfg.min_solidity_box - 0.05
    )

    # Cone-triangle: 3-5 vertices on approxPolyDP.
    is_cone_triangle = (
        cfg.cone_triangle_vertices[0] <= n_vert <= cfg.cone_triangle_vertices[1]
        and solidity >= cfg.min_solidity_box - 0.10
        and aspect <= 0.95  # rules out near-square boxes that triangulate to 4 verts
        and rectangularity < cfg.box_rectangularity_min  # avoid box overlap
    )

    if is_box and not (is_cone_circle and circularity > 0.85):
        # Confidence: how comfortably we're inside the box band.
        conf = min(rectangularity, solidity)
        return "box", conf, metrics

    if is_cone_circle or is_cone_triangle:
        conf = max(circularity, 0.5 if is_cone_triangle else 0.0)
        return "cone", conf, metrics

    # Borderline: be lenient if it's still a tidy convex blob with decent area.
    if solidity >= 0.75 and rectangularity >= 0.6:
        # Lean on aspect to pick the closest class.
        if aspect >= 0.5 and rectangularity >= 0.7:
            return "box", 0.45, metrics
        return "cone", 0.40, metrics

    return "unknown", 0.0, metrics


def extract_blobs(
    pink_mask: np.ndarray, cfg: TransferConfig, verbose: bool = True
) -> List[Blob]:
    img_area = float(pink_mask.shape[0] * pink_mask.shape[1])
    min_area = cfg.min_blob_area_frac * img_area
    max_area = cfg.max_blob_area_frac * img_area

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs: List[Blob] = []
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area or a > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        label, conf, metrics = _classify_blob(c, cfg)

        blobs.append(
            Blob(
                contour=c,
                area=a,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                solidity=metrics.get("solidity", 0.0),
                aspect=metrics.get("aspect", 0.0),
                rectangularity=metrics.get("rectangularity", 0.0),
                circularity=metrics.get("circularity", 0.0),
                n_vertices=int(metrics.get("n_vertices", 0)),
                heuristic_label=label,
                heuristic_confidence=conf,
                final_label=label if label != "unknown" else None,
            )
        )

    _log(
        f"  Found {len(blobs)} candidate blobs after area filter "
        f"[{min_area:.0f}..{max_area:.0f} px]",
        verbose,
    )
    for i, b in enumerate(blobs):
        _log(
            f"    [{i:02d}] area={b.area:7.0f}  solid={b.solidity:.2f}  "
            f"asp={b.aspect:.2f}  rect={b.rectangularity:.2f}  "
            f"circ={b.circularity:.2f}  verts={b.n_vertices}  "
            f"-> {b.heuristic_label} ({b.heuristic_confidence:.2f})",
            verbose,
        )
    return blobs


# ===============================================================================
# Stage 5 — Florence-2 verification (soft import, opt-in)
# ===============================================================================


class Florence2Verifier:
    """Lazy wrapper. Import + load the model only if instantiated."""

    def __init__(self, cfg: TransferConfig, verbose: bool = True):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Florence-2 verification requested but `transformers`/`torch` "
                "are not installed. Install with: "
                "pip install transformers torch pillow\n"
                f"(underlying error: {e})"
            ) from e

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        device = cfg.florence2_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        _log(f"  Loading Florence-2 ({cfg.florence2_model_id}) on {device}...", verbose)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            cfg.florence2_model_id, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.florence2_model_id, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self.cfg = cfg

    def _caption(self, pil_img) -> str:
        import torch

        task = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task, images=pil_img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.lower()

    def verify(self, blob_crop_bgr: np.ndarray) -> Optional[str]:
        """Return the matched expected-shape name or None."""
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(blob_crop_bgr, cv2.COLOR_BGR2RGB))
        caption = self._caption(pil)

        best = None
        for shape in self.cfg.expected_shapes:
            # any keyword from name or descriptions appearing in the caption?
            hits = [shape.name.lower()] + [d.lower() for d in shape.descriptions]
            if any(self._keyword_hit(caption, kw) for kw in hits):
                best = shape.name
                break
        return best

    @staticmethod
    def _keyword_hit(caption: str, phrase: str) -> bool:
        """Very loose phrase match — checks any non-trivial word from the
        phrase appears in the caption."""
        if phrase in caption:
            return True
        # fallback: any meaningful >=4-letter word
        for w in phrase.split():
            if len(w) >= 4 and w in caption:
                return True
        return False


def verify_with_florence2(
    img_bgr: np.ndarray, blobs: List[Blob], cfg: TransferConfig, verbose: bool = True
) -> List[Blob]:
    if not cfg.use_florence2:
        return blobs
    if not blobs:
        return blobs

    verifier = Florence2Verifier(cfg, verbose=verbose)
    pad = cfg.florence2_pad_px
    H, W = img_bgr.shape[:2]
    kept: List[Blob] = []
    for i, b in enumerate(blobs):
        x, y, w, h = b.bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        crop = img_bgr[y0:y1, x0:x1]
        label = verifier.verify(crop)
        if label is None:
            _log(
                f"    [{i:02d}] Florence-2 rejected (heuristic said "
                f"{b.heuristic_label})",
                verbose,
            )
            b.final_label = None
            continue
        b.final_label = label
        _log(
            f"    [{i:02d}] Florence-2: {label} "
            f"(heuristic said {b.heuristic_label})",
            verbose,
        )
        kept.append(b)
    return kept


# ===============================================================================
# Stage 6 — Project onto background
# ===============================================================================


def _detect_background_wall_bbox(
    bg: np.ndarray, cfg: TransferConfig, verbose: bool = True
) -> Tuple[int, int, int, int]:
    """Return the inner bounding box (x, y, w, h) of the brown wall on
    background.png — i.e. the playable arena rectangle."""
    hsv = _hsv(bg)
    lo = np.array(
        [cfg.background_wall_h_lo, cfg.background_wall_s_lo, cfg.background_wall_v_lo],
        dtype=np.uint8,
    )
    hi = np.array(
        [cfg.background_wall_h_hi, 255, cfg.background_wall_v_hi], dtype=np.uint8
    )
    wall = cv2.inRange(hsv, lo, hi)
    pts = cv2.findNonZero(wall)
    if pts is None:
        _log(
            "  [warn] No brown wall found in background — using full image bbox.",
            verbose,
        )
        H, W = bg.shape[:2]
        return 0, 0, W, H
    x, y, w, h = cv2.boundingRect(pts)
    _log(f"  Background wall bbox: x={x} y={y} w={w} h={h}", verbose)
    return x, y, w, h


def _detect_grid_lines_xy(
    img: np.ndarray, cfg: TransferConfig
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
        bg_xs, bg_ys = _detect_grid_lines_xy(bg, cfg)
        _log(
            f"  Grid lines: source ({len(src_xs)}x{len(src_ys)})  "
            f"background ({len(bg_xs)}x{len(bg_ys)})",
            verbose,
        )

        if (
            len(src_xs) >= 2
            and len(bg_xs) >= 2
            and len(src_ys) >= 2
            and len(bg_ys) >= 2
        ):
            # If counts differ, align by trimming the longer one symmetrically.
            def _align(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
                if len(a) == len(b):
                    return a, b
                if len(a) > len(b):
                    drop = len(a) - len(b)
                    left, right = drop // 2, drop - drop // 2
                    return a[left : len(a) - right], b
                drop = len(b) - len(a)
                left, right = drop // 2, drop - drop // 2
                return a, b[left : len(b) - right]

            xa, xb = _align(src_xs, bg_xs)
            ya, yb = _align(src_ys, bg_ys)

            def _map_grid(px: float, py: float) -> Tuple[int, int]:
                qx = _interp_pos(px, xa, xb)
                qy = _interp_pos(py, ya, yb)
                return int(round(qx)), int(round(qy))

            map_grid_fn = _map_grid
        else:
            _log(
                "  [warn] grid mode requested but not enough lines on one side "
                "— falling back to bbox.",
                verbose,
            )

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
        pts_dst = np.array(
            [map_fn(float(p[0]), float(p[1])) for p in pts_src], dtype=np.int32
        ).reshape(-1, 1, 2)

        cv2.drawContours(out, [pts_dst], -1, color, thickness=cv2.FILLED)
        if cfg.draw_outline_px > 0:
            # darker outline
            outline = tuple(int(c * 0.6) for c in color)
            cv2.drawContours(out, [pts_dst], -1, outline, thickness=cfg.draw_outline_px)
        drawn += 1
        _log(f"    [{i:02d}] drew {label} blob ({len(pts_src)} pts)", verbose)

    _log(f"  Drew {drawn} obstacle(s) onto background.", verbose)
    return out


# ===============================================================================
# Debug image helpers
# ===============================================================================


def _save_debug(
    path: Optional[str],
    name: str,
    img: np.ndarray,
    debug_dir: Optional[str],
    verbose: bool = True,
):
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
        text = f"#{i} {b.heuristic_label}"
        if b.final_label and b.final_label != b.heuristic_label:
            text += f"->{b.final_label}"
        cv2.putText(
            out,
            text,
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


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
    bg = cv2.imread(background_path)
    if bg is None:
        raise IOError(f"Cannot read background image: {background_path!r}")
    stages["input"] = img.copy()

    _log(sep, verbose)
    _log(
        f"  Input:      {reconstructed_path}  [{img.shape[1]}x{img.shape[0]}]", verbose
    )
    _log(f"  Background: {background_path}  [{bg.shape[1]}x{bg.shape[0]}]", verbose)
    _log(sep, verbose)

    # Stage 1 ------------------------------------------------------------
    if cfg.correct_perspective:
        _log("\n[1/6] De-warping via blue grid lines...", verbose)
        dewarped = dewarp(img, cfg, verbose=verbose)
    else:
        _log("\n[1/6] Perspective correction disabled.", verbose)
        dewarped = img
    stages["dewarped"] = dewarped
    _save_debug(None, "01_dewarped", dewarped, debug_dir, verbose)

    # Stage 2 ------------------------------------------------------------
    _log("\n[2/6] Cropping to blue boundary...", verbose)
    cropped = crop_to_blue(dewarped, cfg, verbose=verbose)
    stages["cropped"] = cropped
    _save_debug(None, "02_cropped", cropped, debug_dir, verbose)

    _log("\n[2b/6] Masking everything outside the green wall hull...", verbose)
    cleaned, hull = mask_outside_wall(cropped, cfg, verbose=verbose)
    stages["wall_masked"] = cleaned
    _save_debug(None, "03_wall_masked", cleaned, debug_dir, verbose)

    # Stage 3 ------------------------------------------------------------
    _log("\n[3/6] Building pink & green masks with closing...", verbose)
    pink, green = build_clean_masks(cleaned, cfg, verbose=verbose)
    stages["pink_mask"] = pink
    stages["green_mask"] = green
    _save_debug(
        None, "04_pink_mask", _mask_to_bgr(pink, (255, 0, 255)), debug_dir, verbose
    )
    _save_debug(
        None, "05_green_mask", _mask_to_bgr(green, (0, 255, 0)), debug_dir, verbose
    )

    # Stage 4 ------------------------------------------------------------
    _log("\n[4/6] Extracting blobs and classifying heuristically...", verbose)
    blobs = extract_blobs(pink, cfg, verbose=verbose)

    # Stage 5 ------------------------------------------------------------
    if cfg.use_florence2:
        _log("\n[5/6] Verifying every blob with Florence-2...", verbose)
        blobs = verify_with_florence2(cleaned, blobs, cfg, verbose=verbose)
    else:
        _log("\n[5/6] Florence-2 disabled — using heuristic labels.", verbose)

    overlay = _blob_overlay(cleaned, blobs)
    stages["blob_overlay"] = overlay
    _save_debug(None, "06_blob_overlay", overlay, debug_dir, verbose)

    # Stage 6 ------------------------------------------------------------
    _log(
        f"\n[6/6] Projecting blobs onto background " f"(mode={cfg.project_mode})...",
        verbose,
    )
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
    ap.add_argument(
        "--background",
        "-b",
        required=True,
        help="Background template image to draw onto",
    )
    ap.add_argument(
        "--out", "-o", default="result.png", help="Path for the final composited image"
    )
    ap.add_argument("--debug-dir", help="If set, write per-stage debug images here")

    # Stage 1
    ap.add_argument(
        "--no-dewarp",
        action="store_true",
        help="Skip Stage 1 (blue-grid perspective correction)",
    )

    # Stage 3
    ap.add_argument(
        "--close-iters",
        type=int,
        default=3,
        help="MORPH_CLOSE iterations on pink/green masks (default 3)",
    )
    ap.add_argument(
        "--morph-kernel", type=int, default=5, help="Closing kernel size (default 5)"
    )

    # Stage 4
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    ap.add_argument("--max-area-frac", type=float, default=0.25)

    # Stage 5
    ap.add_argument(
        "--use-florence2",
        action="store_true",
        help="Verify every blob with Florence-2 (needs transformers)",
    )
    ap.add_argument("--florence2-model", default="microsoft/Florence-2-base")
    ap.add_argument(
        "--florence2-device", default="auto", choices=["auto", "cpu", "cuda"]
    )

    # Stage 6
    ap.add_argument(
        "--project-mode",
        default="bbox",
        choices=["bbox", "grid"],
        help="bbox = simple scale-and-shift to background wall bbox; "
        "grid = piecewise-linear via detected blue grid lines.",
    )
    ap.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Draw blobs the heuristic couldn't classify "
        "(gray) instead of dropping them.",
    )

    ap.add_argument("--quiet", action="store_true")
    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    cfg = TransferConfig(
        correct_perspective=not args.no_dewarp,
        close_iterations=args.close_iters,
        morph_kernel=args.morph_kernel,
        min_blob_area_frac=args.min_area_frac,
        max_blob_area_frac=args.max_area_frac,
        use_florence2=args.use_florence2,
        florence2_model_id=args.florence2_model,
        florence2_device=args.florence2_device,
        project_mode=args.project_mode,
        drop_unknown=not args.keep_unknown,
    )

    final, _stages = run_pipeline(
        args.image,
        args.background,
        cfg,
        debug_dir=args.debug_dir,
        verbose=not args.quiet,
    )
    cv2.imwrite(args.out, final)
    if not args.quiet:
        print(f"\nFinal image written -> {args.out}")


if __name__ == "__main__":
    main()
