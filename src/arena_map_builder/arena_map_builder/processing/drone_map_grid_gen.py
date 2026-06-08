"""
video_map_reconstruct.py
─────────────────────────────────────────────────────────────────────────────
Drone video → top-down map reconstruction.

Memory model
────────────
Two sources of peak memory in the original version are fixed here:

  1. extract_frames() buffered every filtered frame before stitching started.
     Replaced by stream_frames(), a generator that yields one frame at a time.
     reconstruct_from_video() and MapReconstructor.add_video() both consume
     this generator directly, so only ONE decoded frame lives in RAM at once
     (plus the canvas).

  2. _warp_and_blend() called cv2.warpPerspective(..., (canvas_w, canvas_h)),
     allocating a full-canvas-sized scratch buffer per frame.  As the canvas
     grows (e.g. 8 000 × 6 000 px) that is ~140 MB per frame just for the
     temporary.  Replaced by ROI-based warping: the frame is warped only into
     its bounding-box footprint on the canvas (frame-sized, not canvas-sized),
     then blended directly into the canvas slice.

  3. ReconstructConfig.processing_scale lets you halve or quarter the
     resolution of all incoming frames before processing.  Half resolution
     = 4× smaller canvas and warp buffers, at the cost of output detail.

Grid intersection alignment
───────────────────────────
The arena's black background with a blue tape grid provides highly reliable
geometric landmarks.  A two-stage alignment pipeline uses them:

  Stage 1: SIFT-based similarity RANSAC (existing pipeline) → H_rough
  Stage 2: H_rough predicts where each grid intersection in the current frame
           maps into the reference frame; nearest-neighbour matching (with
           a configurable distance cap) produces sub-pixel accurate
           correspondences that are combined with the Stage-1 SIFT inliers
           for a tighter RANSAC refitting.

  When fewer than grid_min_intersections matches are found, Stage 2 is
  skipped transparently and Stage 1's result is used unmodified.

Public surface
──────────────
    from video_map_reconstruct import (
        ExtractionConfig, ReconstructConfig,
        stream_frames, extract_frames,   # generator and list variants
        MapReconstructor,
        reconstruct_from_video,
    )

Quick-start
───────────
    # Streaming (low memory):
    result = reconstruct_from_video("flight.mp4", output_shape=(3000, 3000))

    # Manual control:
    rec = MapReconstructor()
    rec.add_video("flight.mp4")
    map_img = rec.get_map(output_shape=(3000, 3000))
"""

import os
import cv2
import numpy as np
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

# Global-consistency back-end (fiducial loop closure + pose-graph optimisation).
# Imported flexibly so the module works both as a ROS package and standalone.
try:
    from .fiducials import FiducialDetector, FiducialConfig
    from .stitch_graph import StitchGraph, StitchGraphConfig
except ImportError:  # running as a plain script / tests
    from fiducials import FiducialDetector, FiducialConfig
    from stitch_graph import StitchGraph, StitchGraphConfig

# ── GPU warp backend (checked once at import time) ────────────────────────────
try:
    _CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except (cv2.error, AttributeError):
    _CUDA_AVAILABLE = False
print(f"[drone_map] warp backend: {'CUDA (GPU)' if _CUDA_AVAILABLE else 'CPU'}")

# ══════════════════════════════════════════════════════════════════════════════
# Shared low-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_detector():
    """SIFT with generous feature budget; ORB fallback for older OpenCV builds."""
    try:
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    except AttributeError:
        return cv2.ORB_create(nfeatures=8000), cv2.NORM_HAMMING


def _kp_des(detector, img: np.ndarray, mask: Optional[np.ndarray] = None):
    """Detect keypoints + descriptors on `img`, optionally restricted by `mask`.

    `mask` is a uint8 array the same H×W as img where non-zero pixels are
    eligible for feature detection.  Use it to exclude regions dominated by
    a repetitive structure (e.g. blue grid tape) from feature detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return detector.detectAndCompute(gray, mask)


def _match(des1, des2, norm, ratio: float = 0.70, mutual: bool = True) -> list:
    """Lowe ratio-test match with optional mutual best-match cross-check.

    On scenes with repetitive structure (regular grids, periodic textures),
    the standard ratio test alone is unreliable: the second-best match is
    often a *different* instance of the same repeated feature at nearly
    identical descriptor distance, so many ambiguous matches pass.

    Mutual cross-check requires the match to be best in *both* directions
    AND pass the ratio test in both directions.  This is much stronger
    than cv2.BFMatcher(crossCheck=True), which omits the ratio test.
    """
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw12 = matcher.knnMatch(des1, des2, k=2)
    good12 = [
        m
        for pair in raw12
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    ]
    if not mutual or not good12:
        return good12

    raw21 = matcher.knnMatch(des2, des1, k=2)
    # Map descriptor-2 index → descriptor-1 index for ratio-passing matches.
    back: dict = {}
    for pair in raw21:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            back[m.queryIdx] = m.trainIdx

    return [m for m in good12 if back.get(m.trainIdx) == m.queryIdx]


def _median_displacement(
    kp1, des1, kp2, des2, norm, diag: float
) -> Optional[float]:
    """Robust median pixel displacement between two keypoint sets, normalised
    by `diag` (the frame diagonal).  Returns None if matching is infeasible.

    Used by the movement gate during frame extraction.  Unlike _match() this
    skips the ratio filter entirely — on repetitive scenes the ratio test
    discards almost every match, leaving the motion estimate undefined.  The
    *median* of nearest-neighbour displacements is naturally robust to the
    ambiguous off-by-one-cell matches that result from repeated structure,
    so it gives a reliable static/jerk estimate even there.
    """
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(des1, des2, k=1)
    matches = [pair[0] for pair in raw if len(pair) == 1]
    if len(matches) < 8:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return float(np.median(np.linalg.norm(pts1 - pts2, axis=1))) / diag


def _keypoint_spread_ok(
    kp: list,
    frame_w: int,
    frame_h: int,
    bins: int = 8,
    min_filled: int = 12,
) -> Tuple[bool, str]:
    """Check that keypoints are spatially well-distributed across the frame.

    On scenes dominated by a single repeated structure (e.g. a grid), feature
    detectors fire predominantly on that structure, leaving large regions of
    the frame with no informative features.  Even after RANSAC, such frames
    produce alignment estimates that are highly anisotropic (well-constrained
    along one axis, badly under-constrained along the orthogonal).

    Simple spatial-spread test: bin keypoints into a bins×bins grid over the
    frame and require at least `min_filled` distinct bins non-empty.  With
    bins=8 (64 cells total) and min_filled=12, the test demands ~19% spatial
    coverage — generous in the normal case, aggressive only on truly
    degenerate frames.
    """
    if len(kp) < min_filled:
        return False, f"too_few_kp={len(kp)}"

    if isinstance(kp, np.ndarray):
        xs, ys = kp[:, 0].astype(np.float32), kp[:, 1].astype(np.float32)
    else:
        xs = np.fromiter((p.pt[0] for p in kp), dtype=np.float32, count=len(kp))
        ys = np.fromiter((p.pt[1] for p in kp), dtype=np.float32, count=len(kp))

    bx = np.clip((xs * bins / frame_w).astype(np.int32), 0, bins - 1)
    by = np.clip((ys * bins / frame_h).astype(np.int32), 0, bins - 1)
    keys = by.astype(np.int64) * bins + bx.astype(np.int64)

    filled = int(np.unique(keys).size)
    if filled < min_filled:
        return False, f"spread={filled}<{min_filled}"
    return True, "ok"


def _make_feature_mask(
    img: np.ndarray,
    exclude_hsv: list,
    dilate_px: int = 5,
) -> Optional[np.ndarray]:
    """Build a uint8 mask for cv2.detectAndCompute that EXCLUDES pixels
    matching any of the given HSV ranges (passed as ColorRangeMask objects).

    Use case: in scenes dominated by a repetitive structure (e.g. blue grid
    tape), feature detectors fire predominantly on that structure, producing
    keypoints whose descriptors are near-duplicates of each other.  Excluding
    the dominant structure from feature detection forces the detector onto
    the unique, informative parts of the scene (object edges, corners,
    natural texture), yielding fewer but much more discriminative keypoints.

    The exclusion mask is dilated by `dilate_px` so keypoints don't latch
    onto the *edges* of the excluded structure — those edges still inherit
    grid-locked positions and would re-introduce the ambiguity problem.

    Returns None if `exclude_hsv` is empty (caller should pass mask=None to
    detectAndCompute, equivalent to no masking).
    """
    if not exclude_hsv:
        return None

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    exclude = np.zeros(img.shape[:2], dtype=np.uint8)
    for cm in exclude_hsv:
        lo = np.array([cm.h_lo, cm.s_lo, cm.v_lo], dtype=np.uint8)
        hi = np.array([cm.h_hi, cm.s_hi, cm.v_hi], dtype=np.uint8)
        exclude |= cv2.inRange(img_hsv, lo, hi)
    del img_hsv

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        exclude = cv2.dilate(exclude, kernel)

    # Mask convention for cv2.detectAndCompute: non-zero pixels are eligible.
    return cv2.bitwise_not(exclude)


# ══════════════════════════════════════════════════════════════════════════════
# Blue grid intersection detection
# ══════════════════════════════════════════════════════════════════════════════


def _line_intersection_pt(l1, l2) -> Optional[Tuple[float, float]]:
    """Analytical intersection of two infinite lines, each given as (x1,y1,x2,y2).
    Returns (x, y) or None if lines are parallel."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _cluster_hough_lines(lines, axis: str = "y", min_gap: int = 50) -> list:
    """Collapse many Hough fragments per tape strip into one representative line.

    For horizontal strips (axis='y') clusters by Y midpoint; for vertical
    strips (axis='x') clusters by X midpoint.  Consecutive midpoints more
    than min_gap pixels apart are considered different tape strips.
    """
    if not lines:
        return []

    mids = np.array([
        (l[0] + l[2]) / 2 if axis == "x" else (l[1] + l[3]) / 2
        for l in lines
    ])
    order = np.argsort(mids)
    sorted_lines = [lines[i] for i in order]
    sorted_mids = mids[order]

    clusters, cluster = [], [sorted_lines[0]]
    for i in range(1, len(sorted_lines)):
        if sorted_mids[i] - sorted_mids[i - 1] >= min_gap:
            clusters.append(cluster)
            cluster = []
        cluster.append(sorted_lines[i])
    clusters.append(cluster)

    result = []
    for cl in clusters:
        xs = [p for l in cl for p in (l[0], l[2])]
        ys = [p for l in cl for p in (l[1], l[3])]
        if axis == "y":
            result.append((min(xs), int(np.mean(ys)), max(xs), int(np.mean(ys))))
        else:
            result.append((int(np.mean(xs)), min(ys), int(np.mean(xs)), max(ys)))
    return result


def detect_blue_grid_intersections(
    img: np.ndarray,
    h_lo: int = 90,
    h_hi: int = 130,
    s_lo: int = 50,
    v_lo: int = 50,
    hough_threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 25,
    cluster_min_gap: int = 50,
    dedup_radius: float = 20.0,
) -> np.ndarray:
    """Detect every grid intersection of the blue tape in `img`.

    Pipeline
    ────────
    1. HSV colour mask   — isolate blue pixels
    2. Morphological clean-up (close gaps, remove noise)
    3. Canny edge detection on the mask
    4. Probabilistic Hough to find line segments
    5. Separate into horizontal / vertical segments
    6. Cluster fragments per tape strip → one line per strip
    7. Compute all H×V intersection points analytically
    8. Deduplicate near-coincident points

    Parameters
    ──────────
    h_lo/h_hi   : Hue range in OpenCV convention (0–179).  Default 90–130
                  covers most blue tapes under typical indoor lighting.
    s_lo / v_lo : Minimum saturation and brightness (0–255) to exclude
                  dark shadows and washed-out near-white areas.
    hough_threshold : Minimum Hough accumulator votes.
    min_line_length  : Discard Hough segments shorter than this (px).
    max_line_gap     : Bridge intra-segment gaps up to this width (px).
    cluster_min_gap  : Minimum distance (px) between tape-strip midpoints
                       before they are treated as separate strips.
    dedup_radius     : Intersection points within this radius (px) of an
                       already-kept point are discarded as duplicates.

    Returns
    ───────
    Float32 array of shape (N, 2) with columns [x, y].
    Empty array (shape (0, 2)) when no intersections are found.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── 1. Blue colour mask ──────────────────────────────────────────────────
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255,  255 ], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    del hsv

    # ── 2. Morphological clean-up ────────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── 3. Edge detection ────────────────────────────────────────────────────
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # ── 4. Probabilistic Hough ───────────────────────────────────────────────
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return np.empty((0, 2), dtype=np.float32)
    lines = lines[:, 0, :]   # (N, 4)

    # ── 5. Separate horizontal / vertical ────────────────────────────────────
    horizontal, vertical = [], []
    for x1, y1, x2, y2 in lines:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            horizontal.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            vertical.append((x1, y1, x2, y2))

    # ── 6. Cluster into one line per tape strip ──────────────────────────────
    h_merged = _cluster_hough_lines(horizontal, axis="y", min_gap=cluster_min_gap)
    v_merged = _cluster_hough_lines(vertical,   axis="x", min_gap=cluster_min_gap)

    if not h_merged or not v_merged:
        return np.empty((0, 2), dtype=np.float32)

    # ── 7. Compute H×V intersections ─────────────────────────────────────────
    # Allow a 30-px margin to catch intersections slightly off-screen.
    margin = 30
    raw = []
    for hl in h_merged:
        for vl in v_merged:
            pt = _line_intersection_pt(hl, vl)
            if pt is None:
                continue
            x, y = pt
            if -margin <= x <= w + margin and -margin <= y <= h + margin:
                raw.append((x, y))

    if not raw:
        return np.empty((0, 2), dtype=np.float32)

    # ── 8. Deduplicate near-coincident points ─────────────────────────────────
    kept: List[Tuple[float, float]] = []
    for p in raw:
        if all(
            np.hypot(p[0] - q[0], p[1] - q[1]) > dedup_radius for q in kept
        ):
            kept.append(p)

    return np.array(kept, dtype=np.float32)  # shape (N, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Grid intersection correspondence matching
# ══════════════════════════════════════════════════════════════════════════════


def _match_grid_intersections(
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    H_rough: np.ndarray,
    max_dist: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match grid intersection points from two frames using H_rough as a prior.

    H_rough maps current-frame pixel coordinates into reference-frame pixel
    coordinates (the same convention used by _pairwise_H).  Each intersection
    in the current frame is projected through H_rough; the nearest intersection
    in the reference frame is accepted as a match if its distance after
    projection is within `max_dist` pixels.  One-to-one assignment is enforced
    (each reference intersection can be claimed by at most one current point).

    Parameters
    ──────────
    pts_ref  : (N, 2) float32 — intersection pixel coords in the reference frame
    pts_cur  : (M, 2) float32 — intersection pixel coords in the current frame
    H_rough  : 3×3 float64 homography — cur pixel coords → ref pixel coords
    max_dist : Maximum allowable pixel distance after projection (default 25 px).
               Should be ≤ half the expected grid cell spacing in pixels.

    Returns
    ───────
    (matched_ref, matched_cur) — parallel (K, 2) float32 arrays of
    corresponding point pairs.  Both are empty when no matches are found.
    """
    if len(pts_ref) == 0 or len(pts_cur) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Project current frame intersections into reference frame coordinates.
    ones = np.ones((len(pts_cur), 1), dtype=np.float64)
    pts_cur_h = np.hstack([pts_cur.astype(np.float64), ones])  # (M, 3)
    proj_h = (H_rough @ pts_cur_h.T).T                         # (M, 3)
    # Perspective divide; guard against w ≈ 0.
    w_col = proj_h[:, 2:3]
    safe = np.abs(w_col) > 1e-8
    proj = np.where(safe, proj_h[:, :2] / np.where(safe, w_col, 1.0), 1e9)
    proj = proj.astype(np.float32)  # (M, 2) in ref coords

    # Greedy nearest-neighbour assignment with one-to-one constraint.
    # Build a distance matrix (M × N) and iterate in order of increasing dist.
    pts_ref_f = pts_ref.astype(np.float32)
    diffs = proj[:, np.newaxis, :] - pts_ref_f[np.newaxis, :, :]  # (M, N, 2)
    dist_mat = np.linalg.norm(diffs, axis=2)                       # (M, N)

    # Flatten, sort by distance, then assign greedily.
    flat_idx = np.argsort(dist_mat, axis=None)
    used_cur, used_ref = set(), set()
    matched_cur_idx, matched_ref_idx = [], []

    for idx in flat_idx:
        i = idx // len(pts_ref)   # current-frame intersection index
        j = idx  % len(pts_ref)   # reference-frame intersection index
        if dist_mat[i, j] > max_dist:
            break                 # sorted → all remaining are farther
        if i in used_cur or j in used_ref:
            continue
        used_cur.add(i)
        used_ref.add(j)
        matched_cur_idx.append(i)
        matched_ref_idx.append(j)

    if not matched_cur_idx:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return (
        pts_ref_f[matched_ref_idx],     # (K, 2) reference coords
        pts_cur[matched_cur_idx],       # (K, 2) current coords
    )


# ══════════════════════════════════════════════════════════════════════════════
# Homography estimation (pairwise, with optional grid refinement)
# ══════════════════════════════════════════════════════════════════════════════


def _validate_homography(
    H: np.ndarray,
    frame_w: int,
    frame_h: int,
    max_area_ratio: float = 2.0,
) -> Tuple[bool, str]:
    """
    Reject degenerate homographies before they create ray/fan artifacts.

    Three checks, in order of cheapness:

    1. Perspective divide positivity  — H[2,0]*x + H[2,1]*y + H[2,2] must be
       > 0 at all four frame corners.  If it goes ≤ 0 anywhere the warp folds
       through infinity and creates the characteristic swept-ray artifact.

    2. Convexity of warped quad  — the four warped corners must form a convex
       quadrilateral (all cross-products same sign).  A non-convex quad means
       the frame got "twisted" inside out.

    3. Area ratio  — warped area / original area must stay within
       [1/max_area_ratio, max_area_ratio].  Catches extreme zoom artefacts
       that survive the first two checks.
    """
    corners_src = [
        (0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)
    ]

    # ── check 1: positive perspective divide at every corner ─────────────────
    for cx, cy in corners_src:
        w_prime = H[2, 0] * cx + H[2, 1] * cy + H[2, 2]
        if w_prime <= 1e-6:
            return False, f"negative_w_prime={w_prime:.3e}"

    # ── check 2: convexity of the warped quadrilateral ────────────────────────
    pts = np.float32(corners_src).reshape(-1, 1, 2)
    wc  = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    signs = []
    n = len(wc)
    for i in range(n):
        o, a, b = wc[i], wc[(i + 1) % n], wc[(i + 2) % n]
        cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        signs.append(1 if cross > 0 else -1)
    if len(set(signs)) > 1:
        return False, "non_convex_quad"

    # ── check 3: area ratio ───────────────────────────────────────────────────
    x, y = wc[:, 0], wc[:, 1]
    warped_area = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2
    orig_area   = frame_w * frame_h
    ratio = warped_area / (orig_area + 1e-6)
    if not (1.0 / max_area_ratio <= ratio <= max_area_ratio):
        return False, f"area_ratio={ratio:.2f}"

    return True, "ok"


def _validate_composed_H(
    H: np.ndarray,
    frame_w: int,
    frame_h: int,
    canvas_h: int,
    canvas_w: int,
    max_overflow_px: int,
) -> Tuple[bool, str]:
    """
    Validate the COMPOSED homography (ref["H"] @ H_pair) in canvas coordinates.

    Why this is necessary
    ─────────────────────
    _validate_homography() and the centre-point check in _pairwise_H() both
    operate on H_pair alone — the relative transform between the current frame
    and a reference frame.  That transform can look perfectly healthy while the
    composed result ref["H"] @ H_pair is wildly degenerate because accumulated
    numerical drift in ref["H"] amplifies any residual perspective error.

    This function runs AFTER composition so it catches those cases before they
    reach _expand_canvas() (where a degenerate H would request a 200 GB array).

    Checks
    ──────
    1. Positive perspective divide at all four frame corners (no fold-through).
    2. Warped bounding box stays within canvas + max_overflow_px on every side.
       Overflows larger than this indicate drift/degeneracy, not a legitimate
       frame that just clips slightly past the canvas edge.
    """
    corners_src = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]

    # Check 1: perspective divide must be positive at every corner
    for cx, cy in corners_src:
        w_prime = H[2, 0] * cx + H[2, 1] * cy + H[2, 2]
        if w_prime <= 1e-6:
            return False, f"composed_neg_w={w_prime:.3e}"

    # Project corners into canvas space
    pts = np.float32(corners_src).reshape(-1, 1, 2)
    try:
        wc = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    except cv2.error as exc:
        return False, f"composed_transform_error: {exc}"

    # Check 2: bounding box within canvas + overflow tolerance
    x_min, x_max = float(wc[:, 0].min()), float(wc[:, 0].max())
    y_min, y_max = float(wc[:, 1].min()), float(wc[:, 1].max())
    ov = max_overflow_px

    if x_min < -ov:
        return False, f"composed_x_min={x_min:.0f}<-{ov}"
    if y_min < -ov:
        return False, f"composed_y_min={y_min:.0f}<-{ov}"
    if x_max > canvas_w + ov:
        return False, f"composed_x_max={x_max:.0f}>{canvas_w}+{ov}"
    if y_max > canvas_h + ov:
        return False, f"composed_y_max={y_max:.0f}>{canvas_h}+{ov}"

    return True, "ok"


def _pairwise_H(
    feat_ref: tuple,
    feat_cur: tuple,
    norm: int,
    frame_w: int,
    frame_h: int,
    match_ratio: float = 0.70,
    mad_factor: float = 6.0,
    # ── grid-intersection refinement (Stage 2) ────────────────────────────
    grid_pts_ref: Optional[np.ndarray] = None,
    grid_pts_cur: Optional[np.ndarray] = None,
    grid_match_dist: float = 25.0,
    grid_min_matches: int = 4,
    lg_session=None,
) -> tuple:
    """
    Compute H mapping cur → ref coordinate space via similarity-RANSAC.
    Returns (H, n_inliers) or (None, 0) if the homography is unreliable.

    Why a similarity transform, not a full homography
    ─────────────────────────────────────────────────
    Top-down drone footage at roughly constant altitude is fundamentally a
    2D rigid-motion problem (translation + rotation + small uniform scale
    from altitude variation).  That is 4 degrees of freedom.

    Fitting a full 8-DoF projective homography to it gives RANSAC freedom
    to "explain" off-by-one-cell mismatches on repetitive structure (grid
    intersections) with small perspective terms — terms that look locally
    valid in any single frame (convex quad, positive w', area ratio ≈ 1)
    but compound across hundreds of frames into the fan/ray artifacts
    visible in the bad maps.

    cv2.estimateAffinePartial2D gives a 4-DoF similarity (RANSAC variant).
    The 2x3 result is promoted to 3x3 so downstream warpPerspective /
    composition / canvas-expansion code is unchanged.

    MAD pre-filter
    ──────────────
    Before RANSAC, gross outliers are removed using a median-absolute-
    deviation gate on per-match displacement.  On a repetitive grid, a
    wrong "to-different-cell" match has a displacement vector that points
    to a different cell than the true motion — by definition it lies in a
    separate cluster from the inlier displacements, and the MAD gate
    chops it out before RANSAC ever sees it.  This prevents RANSAC from
    promoting a coherent set of *wrong* matches to a winning hypothesis.

    Stage 2: Grid intersection refinement
    ──────────────────────────────────────
    If grid_pts_ref and grid_pts_cur are supplied and enough matches are
    found, a second RANSAC pass refits the similarity on the combined set
    of Stage-1 feature inliers + grid intersection correspondences, using a
    tighter reprojection threshold (2.0 px instead of 3.0 px).

    Grid intersections are subpixel-accurate geometric landmarks that are
    immune to descriptor ambiguity — they either match (within max_dist of
    the projected prediction) or don't, with no false positives.  Adding
    them tightens the rotation and scale estimate, which is where drift
    accumulates most severely across hundreds of frames.  Stage 2 is
    complementary to both SIFT and SP+LG backends.

    The refined result is used only when it produces at least as many
    inliers as Stage 1 and passes all validation checks.
    """
    kp_r, des_r = feat_ref
    kp_c, des_c = feat_cur

    # ── match point extraction ────────────────────────────────────────────
    if lg_session is not None:
        # SP+LG path: des is (kp_norm [1,K,2], descriptors [1,K,256]);
        # kp is pixel-coord array (K,2).
        kn_r, dn_r = des_r
        kn_c, dn_c = des_c
        if kp_r is None or kp_c is None or len(kp_r) < 8 or len(kp_c) < 8:
            return None, 0, {}
        matches, _ = lg_session.run(
            None, {"kpts0": kn_r, "kpts1": kn_c, "desc0": dn_r, "desc1": dn_c}
        )
        n_raw = len(matches)
        if n_raw < 8:
            return None, 0, {}
        pts_r = kp_r[matches[:, 0]].astype(np.float32)
        pts_c = kp_c[matches[:, 1]].astype(np.float32)
    else:
        # Ratio-test path: des is (N, D) descriptor array; kp is either
        # list[cv2.KeyPoint] (SIFT) or ndarray (K, 2) px coords (SuperPoint).
        if des_r is None or des_c is None or len(kp_r) < 8 or len(kp_c) < 8:
            return None, 0, {}
        good = _match(des_r, des_c, norm, ratio=match_ratio, mutual=True)
        n_raw = len(good)
        if n_raw < 12:
            return None, 0, {}
        if isinstance(kp_r, np.ndarray):
            pts_r = kp_r[[m.queryIdx for m in good]].astype(np.float32)
            pts_c = kp_c[[m.trainIdx for m in good]].astype(np.float32)
        else:
            pts_r = np.float32([kp_r[m.queryIdx].pt for m in good])
            pts_c = np.float32([kp_c[m.trainIdx].pt for m in good])

    # ── Stage 1a: median/MAD displacement pre-filter ──────────────────────
    disp = pts_r - pts_c
    med  = np.median(disp, axis=0)
    dev  = np.linalg.norm(disp - med, axis=1)
    mad  = float(np.median(dev)) + 1e-3
    keep = dev < mad_factor * mad
    n_mad = int(keep.sum())
    if n_mad < 12:
        return None, 0, {}
    pts_r = pts_r[keep]
    pts_c = pts_c[keep]

    # ── Stage 1b: similarity-RANSAC (4 DoF: tx, ty, rotation, scale) ─────
    M, mask = cv2.estimateAffinePartial2D(
        pts_c.reshape(-1, 1, 2),
        pts_r.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=50,
    )
    if M is None:
        return None, 0, {}

    n_in = int(mask.sum()) if mask is not None else 0
    if n_in < 8:
        return None, 0, {}

    # Promote 2×3 affine → 3×3 so downstream warpPerspective /
    # composition / canvas-expansion code is unchanged.
    H = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float64)])

    ok, reason = _validate_homography(H, frame_w, frame_h)
    if not ok:
        return None, 0, {}

    cx, cy = frame_w / 2.0, frame_h / 2.0
    mapped = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0][0]
    if abs(mapped[0]) > frame_w * 6 or abs(mapped[1]) > frame_h * 6:
        return None, 0, {}

    stats = {"n_raw": n_raw, "n_mad": n_mad, "n_in1": n_in, "n_grid": 0, "n_in2": 0}

    # ── Stage 2: grid intersection refinement ────────────────────────────
    # Extract feature inlier correspondences to combine with grid matches.
    if (
        grid_pts_ref is not None
        and grid_pts_cur is not None
        and len(grid_pts_ref) >= grid_min_matches
        and len(grid_pts_cur) >= grid_min_matches
    ):
        inlier_bool = mask.ravel().astype(bool)
        feat_inliers_r = pts_r[inlier_bool]   # (n_in, 2) in ref coords
        feat_inliers_c = pts_c[inlier_bool]   # (n_in, 2) in cur coords

        g_ref, g_cur = _match_grid_intersections(
            grid_pts_ref, grid_pts_cur, H, max_dist=grid_match_dist
        )
        stats["n_grid"] = len(g_ref)

        if len(g_ref) >= grid_min_matches:
            # Combine feature inliers with grid intersection matches.
            combined_r = np.vstack([feat_inliers_r, g_ref])
            combined_c = np.vstack([feat_inliers_c, g_cur])

            # Refit with a tighter threshold — grid matches are accurate
            # enough that 2.0 px is achievable without over-rejecting.
            M2, mask2 = cv2.estimateAffinePartial2D(
                combined_c.reshape(-1, 1, 2),
                combined_r.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=5000,
                confidence=0.999,
                refineIters=100,
            )

            if M2 is not None and mask2 is not None:
                n_in2 = int(mask2.sum())
                stats["n_in2"] = n_in2
                if n_in2 >= n_in:   # only upgrade if at least as good
                    H2 = np.vstack([M2, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
                    ok2, _ = _validate_homography(H2, frame_w, frame_h)
                    mapped2 = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H2)[0][0]
                    centre_ok = (
                        abs(mapped2[0]) <= frame_w * 6
                        and abs(mapped2[1]) <= frame_h * 6
                    )
                    if ok2 and centre_ok:
                        H    = H2
                        n_in = n_in2

    return H, n_in, stats


# ══════════════════════════════════════════════════════════════════════════════
# Frame quality assessment
# ══════════════════════════════════════════════════════════════════════════════


def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance – higher = sharper.
    CV_32F uses half the memory of CV_64F with no meaningful change in the
    variance score used for blurry-frame rejection."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    score = float(lap.var())
    del lap
    return score


def _codec_artifact_ratio(gray: np.ndarray) -> float:
    """
    Estimate DCT 8×8 block-artifact severity.
    Compares mean absolute differences at 8-pixel-spaced block boundaries
    (both vertical AND horizontal) to the overall mean neighbour-difference.

    Symmetric in both axes since H.264/H.265 block artifacts appear on both
    column and row boundaries — checking only columns missed half the cases
    and gave inconsistent results depending on frame orientation.
    """
    gi = gray.astype(np.int16)

    # Vertical block boundaries (column-direction differences)
    diff_col = np.abs(np.diff(gi, axis=1))
    mean_col = float(diff_col.mean()) + 1e-6
    cols     = np.arange(7, gray.shape[1] - 1, 8)
    r_col    = float(diff_col[:, cols].mean() / mean_col) if cols.size else 1.0

    # Horizontal block boundaries (row-direction differences)
    diff_row = np.abs(np.diff(gi, axis=0))
    mean_row = float(diff_row.mean()) + 1e-6
    rows     = np.arange(7, gray.shape[0] - 1, 8)
    r_row    = float(diff_row[rows, :].mean() / mean_row) if rows.size else 1.0

    return 0.5 * (r_col + r_row)


def _assess_frame(
    img: np.ndarray,
    blur_thresh: float,
    artifact_thresh: float,
    gray: Optional[np.ndarray] = None,
    lo_brightness: float = 15.0,
    hi_brightness: float = 240.0,
) -> Tuple[bool, str]:
    """Return (ok, reason_string). reason is 'ok' when the frame passes.

    Pass a pre-computed grayscale image in `gray` to avoid a redundant
    BGR→gray conversion when the caller already has one (stream_frames does).
    """
    if gray is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    mean_bright = float(gray.mean())
    if not (lo_brightness < mean_bright < hi_brightness):
        return False, f"brightness={mean_bright:.1f}"

    blur = _blur_score(gray)
    if blur < blur_thresh:
        return False, f"blur={blur:.1f}<{blur_thresh}"

    artifact = _codec_artifact_ratio(gray)
    if artifact > artifact_thresh:
        return False, f"artifact_ratio={artifact:.2f}>{artifact_thresh}"

    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction  –  generator (memory-efficient) and list variants
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExtractionConfig:
    """Tunable parameters for frame extraction and quality filtering."""

    target_fps: float = 5.0
    """Frames per second to pull from the video."""

    blur_thresh: float = 60.0
    """Minimum Laplacian variance; frames below this are blurry."""

    artifact_thresh: float = 2.0
    """Maximum DCT block artifact ratio; frames above this are likely corrupt."""

    min_movement: float = 0.015
    """Skip if mean feature displacement (normalised by diagonal) < this."""

    max_movement: float = 0.55
    """Drop if mean feature displacement > this (jerk / tracking failure)."""

    static_pixel_thresh: float = 3.0
    """
    Fallback static-frame detector used when feature matching cannot determine
    movement (fewer than 8 good feature matches — typically over low-texture
    or featureless floor regions).

    The current frame is downsampled to a thumbnail and compared against the
    last yielded frame using mean absolute pixel difference (MAD).  If the MAD
    is below this threshold the frame is considered static and skipped, even
    though the feature-based movement score could not be computed.

    Value is in [0, 255] grayscale units.
      2–4  : tight — skips frames with very small lighting flicker (default: 3)
      5–10 : loose — only skips near-identical frames
      0    : disables the fallback entirely (restores original bypass behaviour)

    This closes the gap where low-texture sections bypassed the movement gate
    entirely, causing every sampled frame to be yielded and placed regardless
    of whether the drone was actually moving.
    """


def stream_frames(
    video_path: str,
    cfg: Optional[ExtractionConfig] = None,
    verbose: bool = True,
) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields filtered BGR frames one at a time from `video_path`.

    Because frames are yielded rather than collected into a list, only the
    current frame (plus the caller's canvas) needs to be live in RAM.
    All filtering logic is identical to the original extract_frames().
    """
    cfg = cfg or ExtractionConfig()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path!r}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / cfg.target_fps)))

    if verbose:
        print(
            f"  source: {src_fps:.1f} fps, {n_total} frames  →  "
            f"sampling 1 in {step} frames (~{cfg.target_fps:.1f} fps target)"
        )

    detector, norm = _make_detector()
    prev_kp   = prev_des  = None
    prev_thumb: Optional[np.ndarray] = None   # thumbnail of last yielded frame
    _THUMB_W, _THUMB_H = 160, 90              # ~1/12 linear scale, negligible RAM
    stats = dict(sampled=0, kept=0, quality=0, movement=0)

    frame_idx = -1
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % step != 0:
                continue
            stats["sampled"] += 1

            # ── convert to gray once — reused for quality gate AND keypoints ──
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ── quality gate ───────────────────────────────────────────────
            ok, reason = _assess_frame(img, cfg.blur_thresh, cfg.artifact_thresh,
                                       gray=gray)
            if not ok:
                stats["quality"] += 1
                del gray
                if verbose:
                    print(f"  [drop:quality]  frame {frame_idx:5d}: {reason}")
                continue

            # ── movement gate ──────────────────────────────────────────────
            kp, des = detector.detectAndCompute(gray, None)
            del gray  # done with grayscale; img (BGR) is what gets yielded

            feature_decision_made = False

            if prev_des is not None and des is not None and len(kp) >= 8:
                # Robust median-displacement estimator: skips the ratio test
                # because on repetitive scenes the ratio filter discards most
                # matches, leaving the motion estimate undefined.  The median
                # of nearest-neighbour displacements is naturally robust to
                # the ambiguous matches that result from grid repetition.
                diag = math.hypot(img.shape[1], img.shape[0])
                mv = _median_displacement(prev_kp, prev_des, kp, des, norm, diag)
                if mv is not None:
                    feature_decision_made = True
                    if mv < cfg.min_movement:
                        stats["movement"] += 1
                        if verbose:
                            print(
                                f"  [skip:static]   frame {frame_idx:5d}: "
                                f"mv={mv:.4f}"
                            )
                        continue
                    if mv > cfg.max_movement:
                        stats["movement"] += 1
                        if verbose:
                            print(
                                f"  [drop:jerk]     frame {frame_idx:5d}: "
                                f"mv={mv:.4f}"
                            )
                        continue

            # ── pixel-diff fallback (low-texture / feature-match failure) ──
            # When feature matching cannot determine motion (< 8 good matches),
            # fall back to a fast thumbnail pixel difference against the last
            # yielded frame.  This closes the bypass where every frame in a
            # low-texture hover section was yielded unconditionally.
            if (not feature_decision_made
                    and cfg.static_pixel_thresh > 0
                    and prev_thumb is not None):
                thumb = cv2.resize(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    (_THUMB_W, _THUMB_H),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)
                mad = float(np.abs(thumb - prev_thumb).mean())
                del thumb
                if mad < cfg.static_pixel_thresh:
                    stats["movement"] += 1
                    if verbose:
                        print(
                            f"  [skip:pixel]    frame {frame_idx:5d}: "
                            f"mad={mad:.2f}<{cfg.static_pixel_thresh} "
                            f"(no features)"
                        )
                    continue

            prev_kp, prev_des = kp, des
            prev_thumb = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                (_THUMB_W, _THUMB_H),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            stats["kept"] += 1

            if verbose and stats["kept"] % 30 == 0:
                pct = 100 * frame_idx / max(n_total, 1)
                print(f"  … {stats['kept']} frames kept  ({pct:.0f}% of video)")

            yield img  # ← only one frame lives in RAM at this point

    finally:
        cap.release()
        if verbose:
            print(f"  Extraction done → {stats}")


def extract_frames(
    video_path: str,
    cfg: Optional[ExtractionConfig] = None,
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Collect all filtered frames into a list (convenience / backward-compat).

    For long videos prefer stream_frames() or MapReconstructor.add_video()
    to avoid loading the full frame set into RAM at once.
    """
    return list(stream_frames(video_path, cfg, verbose))


class OnlineFrameGate:
    """Push-model equivalent of stream_frames()'s per-frame filtering.

    stream_frames() is a PULL generator over a video FILE, and its quality +
    movement gates are embedded in that loop (they compare consecutive frames).
    For online stitching the frames arrive one at a time from a live stream, so
    this class applies the SAME gates in a PUSH model: feed each incoming BGR
    frame to accept(); it returns True when the frame should be placed onto the
    map, and False when it is throttled (target_fps), low quality, static, or a
    jerk.  Per-frame state (previous keypoints / thumbnail) is held internally.

    The saved-video path (stream_frames / add_video) is intentionally left
    untouched; this is an additive, parallel intake used only by online mode.
    Reuses the same module-level gate helpers so behaviour matches.
    """

    _THUMB_W, _THUMB_H = 160, 90   # mirror stream_frames' thumbnail size

    def __init__(self, cfg: Optional[ExtractionConfig] = None):
        self.cfg = cfg or ExtractionConfig()
        self.detector, self.norm = _make_detector()
        self._prev_kp = None
        self._prev_des = None
        self._prev_thumb: Optional[np.ndarray] = None
        self._last_considered: Optional[float] = None
        self.stats = dict(seen=0, throttled=0, quality=0, movement=0, accepted=0)

    def accept(self, img: np.ndarray, now_sec: float) -> bool:
        """Return True if `img` passes the gates and should be placed.

        `now_sec` is a monotonic timestamp (seconds) used for the target_fps
        throttle, which bounds how often the (relatively expensive) gates run —
        the live camera publishes far faster than target_fps.
        """
        self.stats["seen"] += 1

        # ── target_fps throttle (replaces stream_frames' 1-in-N subsampling) ──
        if self.cfg.target_fps > 0 and self._last_considered is not None:
            if (now_sec - self._last_considered) < (1.0 / self.cfg.target_fps):
                self.stats["throttled"] += 1
                return False
        self._last_considered = now_sec

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # ── quality gate ──────────────────────────────────────────────────────
        ok, _ = _assess_frame(
            img, self.cfg.blur_thresh, self.cfg.artifact_thresh, gray=gray)
        if not ok:
            self.stats["quality"] += 1
            return False

        # ── movement gate (feature displacement; pixel-diff fallback) ─────────
        kp, des = self.detector.detectAndCompute(gray, None)
        feature_decision_made = False
        if self._prev_des is not None and des is not None and len(kp) >= 8:
            diag = math.hypot(img.shape[1], img.shape[0])
            mv = _median_displacement(
                self._prev_kp, self._prev_des, kp, des, self.norm, diag)
            if mv is not None:
                feature_decision_made = True
                if mv < self.cfg.min_movement or mv > self.cfg.max_movement:
                    self.stats["movement"] += 1
                    return False

        if (not feature_decision_made
                and self.cfg.static_pixel_thresh > 0
                and self._prev_thumb is not None):
            thumb = cv2.resize(
                gray, (self._THUMB_W, self._THUMB_H),
                interpolation=cv2.INTER_AREA).astype(np.float32)
            mad = float(np.abs(thumb - self._prev_thumb).mean())
            if mad < self.cfg.static_pixel_thresh:
                self.stats["movement"] += 1
                return False

        # ── accept: advance state ─────────────────────────────────────────────
        self._prev_kp, self._prev_des = kp, des
        self._prev_thumb = cv2.resize(
            gray, (self._THUMB_W, self._THUMB_H),
            interpolation=cv2.INTER_AREA).astype(np.float32)
        self.stats["accepted"] += 1
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Blending utilities
# ══════════════════════════════════════════════════════════════════════════════


def _feather_blend_roi(
    canvas_roi: np.ndarray,
    warped_roi: np.ndarray,
    mask_new: np.ndarray,
    mask_c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Distance-weighted feathering blend operating on pre-extracted ROI arrays.
    Both inputs are the same small region; the full canvas is never touched.

    mask_c : pre-computed boolean coverage mask for canvas_roi.  When None,
             falls back to the pixel-sum heuristic (canvas_roi.sum > 0), which
             misidentifies genuinely black scene pixels as "not yet covered".

    Alpha is derived from the dual distance transform:
        alpha = dist_new / (dist_new + dist_old + eps)
    where dist_new = distance from the new frame's footprint boundary and
    dist_old = distance from the existing coverage boundary.  This seats the
    seam at the iso-depth contour of both footprints — each image is weighted
    by how "deep inside" its own territory the pixel sits — rather than giving
    the new frame higher weight at the center of the overlap zone (the old
    single-distanceTransform approach).
    """
    if mask_c is None:
        mask_c = canvas_roi.sum(axis=2) > 0
    only_new = mask_new & ~mask_c
    overlap = mask_new & mask_c

    result = canvas_roi.copy()
    result[only_new] = warped_roi[only_new]

    if not overlap.any():
        return result

    # Dual distance transform: each source is weighted by how far its pixel
    # sits from its own footprint boundary relative to the other source.
    dist_new = cv2.distanceTransform(mask_new.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    dist_old = cv2.distanceTransform(mask_c.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    alpha = (dist_new / (dist_new + dist_old + 1e-6)).astype(np.float32)
    a3 = alpha[:, :, np.newaxis]

    blended = (
        warped_roi[..., :3].astype(np.float32) * a3
        + canvas_roi[..., :3].astype(np.float32) * (1.0 - a3)
    ).astype(np.uint8)
    result[overlap] = blended[overlap]
    return result


def _laplacian_pyramid_blend(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    Multi-band Laplacian pyramid blend of two same-shape BGR images.
    alpha : float32 H×W, 0 → img_a, 1 → img_b
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    m = alpha[:, :, np.newaxis].astype(np.float32) if alpha.ndim == 2 else alpha

    def _gpyr(img, lvl):
        gp = [img]
        for _ in range(lvl - 1):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def _lpyr(gp):
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1].copy())
        return lp

    lp_a = _lpyr(_gpyr(a, levels))
    lp_b = _lpyr(_gpyr(b, levels))
    gp_m = _gpyr(m, levels)

    blended_lp = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        if gm.shape[:2] != la.shape[:2]:
            gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
        if gm.ndim == 2:
            gm = gm[:, :, np.newaxis]
        blended_lp.append(la + gm * (lb - la))

    img = blended_lp[-1]
    for lvl in reversed(blended_lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(lvl.shape[1], lvl.shape[0])) + lvl

    return np.clip(img, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# SuperPoint + LightGlue ONNX backend helpers
# ══════════════════════════════════════════════════════════════════════════════

def _default_sp_lg_paths():
    """Resolve default SP+LG ONNX paths.

    In a ROS install: share/arena_map_builder/data/models/ (installed by colcon).
    Fallback for standalone use: sibling LightGlue-ONNX-Jetson/weights/ repo.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        models_dir = os.path.join(
            get_package_share_directory("arena_map_builder"), "data", "models"
        )
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.normpath(
            os.path.join(here, "..", "..", "..", "LightGlue-ONNX-Jetson", "weights")
        )
    return (
        os.path.join(models_dir, "superpoint_only.onnx"),
        os.path.join(models_dir, "lightglue_only.onnx"),
    )


def _make_ort_session(path: str, provider: str):
    """Create a single ORT inference session. Lazy-imports onnxruntime."""
    import onnxruntime as ort  # type: ignore

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider.lower() == "cuda"
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _make_sp_lg_sessions(sp_path: str, lg_path: str, provider: str):
    """Create ORT inference sessions for the split SP+LG pipeline."""
    sp_sess = _make_ort_session(sp_path, provider)
    lg_sess = _make_ort_session(lg_path, provider)
    print(
        f"[drone_map] SP+LG sessions loaded  "
        f"provider={sp_sess.get_providers()[0]}  "
        f"sp={os.path.basename(sp_path)}  lg={os.path.basename(lg_path)}"
    )
    return sp_sess, lg_sess


def _sp_extract_feats(
    sp_session, img: np.ndarray, model_h: int, model_w: int
):
    """Extract SuperPoint features from a single BGR frame.

    The exported model has a static batch-2 input shape [2, 1, model_h, model_w],
    so the frame is resized to that shape before inference and the returned
    keypoint pixel coordinates are scaled back to the original frame space.

    Returns
    -------
    kp_px   : (K, 2) float32   keypoint pixel coordinates (x, y) in original frame
    kp_norm : (1, K, 2) float32  keypoints normalised to [-1, 1] for LightGlue
    des     : (1, K, 256) float32  descriptors
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if h != model_h or w != model_w:
        gray = cv2.resize(gray, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    pp = (gray.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]      # (1,1,mH,mW)
    pair = np.concatenate([pp, pp], axis=0)                              # (2,1,mH,mW)
    kp_out, _, des_out = sp_session.run(None, {"images": pair})
    kp_model = kp_out[0]                                                 # (K,2) in model space
    # Scale keypoints back to original frame pixel coordinates.
    kp_px = kp_model * np.array([w / model_w, h / model_h], dtype=np.float32)
    kp_norm = (
        2.0 * kp_px / np.array([w, h], dtype=np.float32) - 1.0
    )[np.newaxis]                                                        # (1,K,2) in [-1,1]
    return kp_px, kp_norm, des_out[0:1]                                  # (K,2),(1,K,2),(1,K,256)


# ══════════════════════════════════════════════════════════════════════════════
# Incremental map reconstruction
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColorRangeMask:
    """
    One HSV color range whose matching pixels are replaced before a frame is
    stitched onto the canvas.

    Parameters
    ──────────
    name        : human-readable label (for logging / debugging)
    h_lo/h_hi   : Hue   range in OpenCV convention  [0, 180]
    s_lo/s_hi   : Saturation range                   [0, 255]
    v_lo/v_hi   : Value (brightness) range           [0, 255]
    replace_bgr : BGR colour written to matching pixels (default: white)

    Pre-built factory helpers
    ─────────────────────────
        ColorRangeMask.yellow()
        ColorRangeMask.brown()
        ColorRangeMask.orange()

    These match the same HSV ranges used by map_to_occupancy.py so the
    stitched canvas has the same colour semantics as the occupancy pipeline.

    Example — mask yellow and brown to white before stitching:
        cfg = ReconstructConfig(
            color_masks=[ColorRangeMask.yellow(), ColorRangeMask.brown()]
        )
    """
    name:        str
    h_lo:        int
    h_hi:        int
    s_lo:        int
    s_hi:        int
    v_lo:        int
    v_hi:        int
    replace_bgr: Tuple[int, int, int] = (255, 0, 255)   # bright pink

    # ── convenience factories ────────────────────────────────────────────────
    @staticmethod
    def yellow(replace_bgr: Tuple[int, int, int] = (255, 255, 255)) -> "ColorRangeMask":
        """Yellow obstacles (H 18–38, high S/V)."""
        return ColorRangeMask("yellow", 18, 38, 80, 255, 80, 255, replace_bgr)

    @staticmethod
    def brown(replace_bgr: Tuple[int, int, int] = (255, 255, 255)) -> "ColorRangeMask":
        """Brown border / cardboard (H 5–25)."""
        return ColorRangeMask("brown", 5, 25, 50, 255, 40, 180, replace_bgr)

    @staticmethod
    def orange(replace_bgr: Tuple[int, int, int] = (255, 255, 255)) -> "ColorRangeMask":
        """Orange cones (H 5–18, high S)."""
        return ColorRangeMask("orange", 5, 18, 120, 255, 80, 255, replace_bgr)

    @staticmethod
    def blue_tape(replace_bgr: Tuple[int, int, int] = (255, 255, 255)) -> "ColorRangeMask":
        """Blue arena grid tape (H 90–125, medium-high S/V).

        Intended for use with ReconstructConfig.feature_exclude_hsv to
        suppress feature detection on the repetitive grid structure.
        Tune the ranges per-arena if your tape colour drifts (lighting,
        wear, or different tape brand).
        """
        return ColorRangeMask("blue_tape", 90, 125, 60, 255, 60, 255, replace_bgr)


def _apply_color_masks(img: np.ndarray, masks: List[ColorRangeMask]) -> np.ndarray:
    """
    Replace pixels matching any of the given HSV color ranges with each
    mask's replacement colour (default: white).

    Returns a copy of `img` with matching pixels recoloured; the original
    is never modified (it is still needed for feature detection in add_frame).

    Memory layout:
      - img       : original frame (caller holds it)
      - result    : one copy of img (this is what we return)
      - img_hsv   : one HSV copy, used only while building masks, then freed
      - per-mask binary mask (uint8, 1 byte/px) — small, freed per iteration

    Peak extra: 2 × frame_size (result + img_hsv).  img_hsv is explicitly
    deleted before return so it is freed as soon as the caller's reference
    to result is the only live object.
    """
    if not masks:
        return img

    result  = img.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for cm in masks:
        lo   = np.array([cm.h_lo, cm.s_lo, cm.v_lo], dtype=np.uint8)
        hi   = np.array([cm.h_hi, cm.s_hi, cm.v_hi], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lo, hi)
        if mask.any():
            result[mask > 0] = cm.replace_bgr
        del mask  # small but free eagerly inside the loop

    del img_hsv  # release the 6 MB HSV copy before returning
    return result


# ── solid BGR fill colours for ArUco markers ─────────────────────────────────
# Distinct from the magenta/green and grid/obstacle colours already used
# elsewhere in the map, so recoloured markers stand out unambiguously.
CYAN_BGR: Tuple[int, int, int] = (255, 255, 0)
RED_BGR:  Tuple[int, int, int] = (0, 0, 255)


def _recolor_aruco_markers(
    img: np.ndarray,
    markers: list,
    color_map: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Replace each detected ArUco marker with a solid BGR colour keyed by its ID.

    Unlike _apply_color_masks (which thresholds on HSV colour), this masks by
    GEOMETRY: each marker's exact 4-corner quad — as returned subpixel-accurate
    by FiducialDetector.detect() — is filled with color_map[marker_id].  This
    sidesteps the black/white abundance problem entirely: only the polygon the
    detector reports as a marker is touched, nothing else.

    Behaviour
    ─────────
    • Markers whose ID is not in color_map are left untouched.
    • Only the exact detected quad is filled (no quiet-zone / margin expansion).
    • `img` is never mutated: a copy is made lazily on the first marker that
      actually matches, so when nothing matches the original array is returned
      unchanged (and stays safe for feature detection on the caller side).

    Returns the (possibly recoloured) image.
    """
    if not color_map or not markers:
        return img

    result: Optional[np.ndarray] = None
    for m in markers:
        bgr = color_map.get(m.marker_id)
        if bgr is None:
            continue
        if result is None:
            result = img.copy()
        quad = np.round(m.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(result, quad, bgr)

    return result if result is not None else img


@dataclass
class ReconstructConfig:
    """Tunable parameters for incremental frame stitching."""

    canvas_margin: int = 2000
    """Initial blank padding (px) around the first frame."""

    min_inliers: int = 10
    """Minimum RANSAC inliers to accept a homography."""

    lookback: int = 4
    """Number of recently placed frames to try matching against."""

    keyframe_interval: int = 10
    """Cache a long-range keyframe every N successfully placed frames."""

    blend_mode: str = "pyramid"
    """
    "feather"  – distance-weighted linear blend  (fast, default)
    "pyramid"  – Laplacian pyramid multi-band     (best quality, slower)
    "flat"     – simple 50/50 average             (debug / benchmark)
    """

    pyramid_levels: int = 4
    """Laplacian pyramid depth; only used when blend_mode='pyramid'."""

    processing_scale: float = 0.5
    """
    Resize every incoming frame by this factor before any processing.

    0.5  → half-resolution  (4× smaller canvas and warp buffers)
    0.25 → quarter-resolution (16×)

    The final map is at the scaled resolution; pass a large output_shape
    to get_map() if you want to upsample it.  This is the fastest single
    knob to turn when the arena is large or the video is high-resolution:

        ReconstructConfig(processing_scale=0.5)
    """

    color_masks: List[ColorRangeMask] = field(default_factory=list)
    """
    List of HSV color ranges to neutralise (replace) in each frame
    *before* it is warped onto the canvas.

    Feature detection for homography estimation still uses the original
    unmasked frame, so alignment quality is not affected.

    Example — mask yellow obstacles and brown border to white:

        ReconstructConfig(
            color_masks=[ColorRangeMask.yellow(), ColorRangeMask.brown()]
        )

    Custom range example (add more hues as needed):

        ReconstructConfig(
            color_masks=[
                ColorRangeMask("lime", h_lo=35, h_hi=75,
                               s_lo=60, s_hi=255, v_lo=60, v_hi=255,
                               replace_bgr=(255, 255, 255)),
            ]
        )
    """

    marker_color_map: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    """
    Map of ArUco marker ID → solid BGR colour to paint that marker before it is
    stitched onto the canvas.  Empty (default) → no marker recolouring at all.

    This is a GEOMETRIC mask, separate from color_masks: a dedicated
    FiducialDetector runs per frame and each listed marker's exact 4-corner
    quad is filled with its colour (no quiet-zone margin).  Markers whose ID is
    not in the map are left untouched.  Detection is independent of
    use_fiducials, so visual recolouring works even with the pose-graph
    fiducial back-end disabled.

    As with color_masks, feature detection / homography estimation still run on
    the original unmasked frame, so alignment quality is unaffected.

    Convenience BGR constants CYAN_BGR and RED_BGR are provided for distinct,
    unused colours.  Example — paint markers 0/1/2 cyan and 10/11 red:

        ReconstructConfig(
            marker_color_map={0: CYAN_BGR, 1: CYAN_BGR, 2: CYAN_BGR,
                              10: RED_BGR, 11: RED_BGR},
        )
    """

    max_keyframes: int = 20
    """
    Maximum number of long-range keyframes kept in memory at any time.

    Keyframes are added every keyframe_interval placed frames and act as
    global re-localisation anchors.  Without a cap they accumulate
    unboundedly (each holds a full SIFT descriptor array of ~2.5 MB).

    When the cap is reached the buffer is thinned to keep an evenly-spaced
    subset of the existing keyframes, preserving temporal coverage while
    bounding memory to max_keyframes × ~2.5 MB ≈ 50 MB at the default.

    Set to 0 to disable the cap (original behaviour, unbounded growth).
    """

    max_canvas_px: int = 8000
    """
    Pre-allocate the canvas at this size (square, in pixels) on the first
    frame instead of starting small and expanding on overflow.

    Why this matters
    ────────────────
    The original _expand_canvas() allocates a brand-new array of the new
    size, copies the old canvas into it, then releases the old one.  During
    that copy BOTH canvases are live simultaneously — a 144 MB canvas
    produces a 288 MB spike every time the drone approaches an edge.

    With pre-allocation the canvas is allocated once at startup and never
    reallocated.  All homographies are expressed in the coordinate space of
    this fixed canvas from the first frame onward, so _expand_canvas()
    becomes a bounds-check only.

    Memory cost: max_canvas_px² × 3 bytes
        6000 px →  ~103 MB  (conservative, small arena)
        8000 px →  ~183 MB  (default, typical indoor arena)
       10000 px →  ~286 MB  (large arena or high-margin scan)

    Set to 0 to disable pre-allocation and revert to the dynamic expansion
    strategy (original behaviour, with expansion spikes).
    """

    # ── alignment-quality knobs (see _match, _pairwise_H) ────────────────────

    match_ratio: float = 0.80
    """Lowe's ratio threshold for inter-frame descriptor matching.

    Lower = stricter (fewer but more reliable matches).  0.70 is a good
    default for the repetitive blue-grid arena; on richer scenes 0.75
    works fine.  Pair with mutual=True (always on inside _match) for
    cross-direction agreement.
    """

    mad_factor: float = 6.0
    """Median-Absolute-Deviation gate factor applied to per-match pixel
    displacement BEFORE RANSAC, inside _pairwise_H.

    Matches whose displacement deviates from the median by more than
    mad_factor × MAD are rejected as gross outliers.  On a repetitive
    grid, off-by-one-cell wrong matches form a separate displacement
    cluster from the true inliers — this filter removes them so RANSAC
    cannot promote them to a winning hypothesis.

    6.0 ≈ 4 sigma equivalent; raise if too aggressive on noisy frames,
    lower (e.g. 4.0) for tighter rejection on very clean scenes.
    """

    feature_exclude_hsv: List[ColorRangeMask] = field(default_factory=list)
    """HSV ranges to EXCLUDE from feature detection (cv2.detectAndCompute mask).

    For arenas dominated by a repetitive structure (e.g. blue grid tape),
    feature detectors fire predominantly on that structure, producing
    keypoints whose descriptors are near-duplicates of each other.
    Excluding the structure forces the detector onto unique scene content
    (object edges, corners, natural texture), yielding fewer but much
    more discriminative keypoints — which is what RANSAC actually needs.

    Different from color_masks: color_masks recolours pixels in the IMAGE
    before stitching (visual-only); feature_exclude_hsv suppresses
    KEYPOINT DETECTION in those regions (alignment-only).  You can use
    both, and they don't have to overlap.

    Example — exclude the blue grid tape from feature detection:
        ReconstructConfig(
            feature_exclude_hsv=[ColorRangeMask.blue_tape()],
        )
    """

    feature_exclude_dilate_px: int = 5
    """Dilation (in pixels) applied to the feature-exclusion mask.

    Without dilation, keypoints latch onto the *edges* of the excluded
    region — but those edges still inherit grid-locked positions and
    re-introduce the ambiguity problem the exclusion was meant to fix.
    Dilating the mask by a few pixels pushes feature detection cleanly
    off the structure.  Tune relative to your line/tape thickness.
    """

    min_keypoint_bins: int = 10
    """Minimum number of distinct 8×8 spatial bins (out of 64) that must
    contain at least one keypoint for a frame to be eligible for placement.

    Catches frames where all keypoints land on a single dominant
    structure (e.g. blue grid tape that wasn't fully excluded, or a frame
    that happens to be aimed at a featureless region).  Such frames would
    yield alignment estimates that are highly anisotropic and prone to
    drift.

    Set to 0 to disable the check.
    """

    # ── blue grid intersection alignment ─────────────────────────────────────

    use_grid_intersections: bool = True
    """Enable the two-stage grid-intersection alignment refinement.

    When True, every frame's blue tape grid intersections are detected once
    during add_frame() and stored alongside the SIFT features in the frame
    registry.  During pairwise matching, the Stage-1 SIFT similarity is used
    as a prior to project and match intersections from the current frame into
    the reference frame; the matched pairs are combined with the SIFT inliers
    for a tighter second RANSAC pass.

    Set to False to disable completely and revert to the original SIFT-only
    pipeline (useful for arenas without a visible grid, or for benchmarking).
    """

    grid_match_dist: float = 25.0
    """Maximum pixel distance (in the reference frame) for a grid intersection
    match to be accepted during Stage-2 refinement.

    After projecting a current-frame intersection through H_rough, it must
    fall within grid_match_dist pixels of a reference-frame intersection to
    be treated as a correspondence.

    Rule of thumb: set to ≤ half the expected grid cell spacing in pixels.
    For a 4×4 m arena with 0.5 m cell spacing viewed from ~2 m altitude, the
    projected cell spacing is roughly 150–200 px, so the default 25 px cap is
    tight enough to prevent cross-cell false matches while forgiving minor
    perspective distortion and quantisation error.

    Raise this if intersections are detected but Stage 2 rarely fires
    (arena far away → small projected cell size).  Lower if you observe
    wrong cross-cell matches contaminating the refined estimate.
    """

    grid_min_intersections: int = 4
    """Minimum number of matched grid intersections required before Stage-2
    refinement is attempted.

    Four points are the minimum for a unique similarity transform (2 DoF per
    point, 4 DoF model), but a slightly higher value (default 4) filters out
    coincidental near-matches from partially visible grids.  Increase to 6–8
    for a stricter gate when the grid is always fully visible.
    """

    grid_hsv_h_lo: int = 90
    """Lower Hue bound (OpenCV 0–179) for the blue tape colour mask used
    during grid intersection detection.  Adjust if your tape reads differently
    under your arena lighting."""

    grid_hsv_h_hi: int = 130
    """Upper Hue bound for the blue tape colour mask."""

    grid_hsv_s_lo: int = 50
    """Minimum Saturation for the blue tape colour mask.  Increase to exclude
    pale / washed-out near-blue regions that aren't actually tape."""

    grid_hsv_v_lo: int = 50
    """Minimum Value (brightness) for the blue tape colour mask.  Increase to
    exclude dark shadows that fall in the blue hue range."""

    # ── feature backend ───────────────────────────────────────────────────────

    feature_extractor: str = "sift"
    """Keypoint extractor.

    "superpoint" — SuperPoint via ONNX Runtime (default). Requires
        onnxruntime-gpu and superpoint_only.onnx in data/models/.
    "sift"       — classic OpenCV SIFT. No extra dependencies.
    """

    feature_matcher: str = "ratio_test"
    """Descriptor matcher.

    "lightglue"  — LightGlue via ONNX Runtime (default). Learned matcher;
        requires onnxruntime-gpu and lightglue_only.onnx in data/models/.
        NOTE: the bundled model was exported for SuperPoint descriptors only;
        pairing with feature_extractor="sift" will raise at init.
    "ratio_test" — BFMatcher + Lowe ratio test + mutual check. Works with
        both extractors (SIFT 128-d and SuperPoint 256-d).
    """

    sp_onnx_path: Optional[str] = None
    """Path to superpoint_only.onnx.

    None (default) resolves automatically: share/arena_map_builder/data/models/
    in a ROS install, or sibling LightGlue-ONNX-Jetson/weights/ for standalone
    use. Only used when feature_extractor == "superpoint".
    """

    lg_onnx_path: Optional[str] = None
    """Path to lightglue_only.onnx. Same auto-resolve logic as sp_onnx_path.
    Only used when feature_matcher == "lightglue"."""

    sp_ort_provider: str = "cuda"
    """ORT execution provider for SP / LG sessions.

    "cuda" — CUDA execution provider (recommended on Jetson, ~72 ms extract).
    "cpu"  — CPU fallback (~350 ms extract, use only for debugging).
    """

    # ── global consistency: fiducial loop closure + pose-graph solve ──────────

    use_pose_graph: bool = True
    """Accumulate a keyframe pose graph during the online pass and run a global
    least-squares solve when the stream stops, then re-render the map from the
    optimised poses. This is what removes inter-pass discontinuities and the
    bowing-grid distortion. The live (streaming) map is unaffected; the clean
    map is produced by finalize()."""

    use_fiducials: bool = True
    """Detect ArUco markers per frame and use repeat sightings of the same ID
    as zero-ambiguity loop-closure landmarks in the pose graph. Markers move
    between runs and aren't surveyed, so they are NOT absolute anchors — but
    within a run a shared ID ties the frames that see it (e.g. the forward and
    backward lawnmower passes)."""

    fiducial_dictionary: str = "DICT_4X4_50"
    """ArUco dictionary name for the arena markers."""

    pg_marker_weight: float = 15.0
    """Weight of marker-corner constraints (subpixel-accurate → trusted most)."""
    pg_odom_weight: float = 1.0
    """Weight of consecutive-keyframe odometry edges (the online chain)."""
    pg_loop_weight: float = 2.0
    """Weight of direct keyframe↔keyframe overlap matches (feature loop closures)."""
    pg_iterations: int = 10
    """Huber IRLS iterations for the global solve."""
    pg_huber_delta: float = 4.0
    """Huber threshold (px) for robustness to bad loop closures."""

    render_cache_dir: Optional[str] = None
    """Directory for the per-frame render cache used by finalize()'s second
    pass. None → a temp dir is created and removed automatically. The cache
    holds one PNG per placed frame so the corrected map can be re-rendered
    without keeping frames in RAM (preserves the streaming memory profile, and
    works for live feeds where re-streaming the source isn't possible)."""


class MapReconstructor:
    """
    Incremental frame-by-frame drone map reconstructor.

    Key memory properties
    ─────────────────────
    • add_frame() accepts ONE frame at a time; no frame list is kept.
    • _warp_and_blend_roi() warps only into the bounding-box footprint of
      the incoming frame (frame-sized temp, not canvas-sized).
    • The canvas grows as needed; its size is proportional to the physical
      area covered, not the number of frames processed.

    Usage
    ─────
        rec = MapReconstructor(cfg)

        # Option A – stream directly from video (most memory-efficient):
        rec.add_video("flight.mp4")

        # Option B – feed frames manually:
        for frame in stream_frames("flight.mp4"):
            rec.add_frame(frame)

        map_img = rec.get_map(output_shape=(3000, 3000))
    """

    def __init__(self, cfg: Optional[ReconstructConfig] = None):
        self.cfg = cfg or ReconstructConfig()

        extractor = self.cfg.feature_extractor
        matcher   = self.cfg.feature_matcher

        if extractor == "sift" and matcher == "lightglue":
            raise ValueError(
                "feature_extractor='sift' + feature_matcher='lightglue' is not "
                "supported: the bundled lightglue_only.onnx was exported for "
                "SuperPoint descriptors only. Use feature_extractor='superpoint' "
                "or feature_matcher='ratio_test'."
            )

        # ── Extractor ────────────────────────────────────────────────────────
        if extractor == "superpoint":
            _sp_default, _ = _default_sp_lg_paths()
            sp_path = self.cfg.sp_onnx_path or _sp_default
            self._sp_session = _make_ort_session(sp_path, self.cfg.sp_ort_provider)
            print(
                f"[drone_map] SP session loaded  "
                f"provider={self._sp_session.get_providers()[0]}  "
                f"model={os.path.basename(sp_path)}"
            )
            self.detector = None
            self.norm = cv2.NORM_L2   # used if matcher == "ratio_test"
            # Derive the model's static spatial dimensions from its input metadata.
            # Fall back to 480×640 if the model reports dynamic dims.
            _sp_shape = self._sp_session.get_inputs()[0].shape  # [2, 1, H, W]
            self._sp_model_h = _sp_shape[2] if isinstance(_sp_shape[2], int) else 480
            self._sp_model_w = _sp_shape[3] if isinstance(_sp_shape[3], int) else 640
            # SP warmup using the actual model dimensions
            _dummy = np.zeros((2, 1, self._sp_model_h, self._sp_model_w), dtype=np.float32)
            kp_w, _, des_w = self._sp_session.run(None, {"images": _dummy})
            print("[drone_map] SP warmup done.")
        else:  # sift
            self.detector, self.norm = _make_detector()
            self._sp_session = None
            kp_w = des_w = None

        # ── Matcher ──────────────────────────────────────────────────────────
        if matcher == "lightglue":
            _, _lg_default = _default_sp_lg_paths()
            lg_path = self.cfg.lg_onnx_path or _lg_default
            self._lg_session = _make_ort_session(lg_path, self.cfg.sp_ort_provider)
            print(
                f"[drone_map] LG session loaded  "
                f"provider={self._lg_session.get_providers()[0]}  "
                f"model={os.path.basename(lg_path)}"
            )
            # LG warmup (needs SP output shapes; only reachable when SP is loaded)
            kn_w = np.zeros((1, kp_w.shape[1], 2), dtype=np.float32)
            dn_w = des_w[0:1]
            self._lg_session.run(
                None, {"kpts0": kn_w, "kpts1": kn_w, "desc0": dn_w, "desc1": dn_w}
            )
            print("[drone_map] LG warmup done.")
        else:  # ratio_test
            self._lg_session = None

        self._canvas: Optional[np.ndarray] = None
        # Boolean coverage mask — True wherever the canvas has been painted.
        # Kept alongside _canvas so genuinely black scene pixels (0, 0, 0) are
        # correctly distinguished from unpainted canvas cells (also 0, 0, 0).
        self._coverage: Optional[np.ndarray] = None

        # Ring buffer of recently placed frames {kp, des, H, grid_pts, kf_id}.
        # deque with a hard maxlen gives O(1) append/drop vs. O(N) list.pop(0).
        self._recent: deque = deque(maxlen=self.cfg.lookback + 2)
        self._keyframes: List[dict] = [] # sparse long-range anchors

        self._n_placed = 0
        self._n_failed = 0

        # Running counters for grid-refinement telemetry.
        self._n_grid_refined = 0
        self._n_grid_skipped = 0

        # Persistent thread pool for parallel candidate matching in _place_frame.
        # Creating a new ThreadPoolExecutor per frame (the original code) spawns
        # and tears down threads at the frame rate; OpenCV RANSAC/matcher DO
        # release the GIL so the parallelism is real, but the OS overhead is not.
        _max_workers = max(1, self.cfg.lookback + self.cfg.max_keyframes)
        self._pool = ThreadPoolExecutor(max_workers=_max_workers)

        # ── global-consistency back-end ──────────────────────────────────────
        self._fiducial = (
            FiducialDetector(FiducialConfig(dictionary=self.cfg.fiducial_dictionary))
            if self.cfg.use_fiducials else None
        )
        # Dedicated detector for visual marker recolouring (marker_color_map).
        # Kept separate from self._fiducial so recolouring works regardless of
        # use_fiducials; instantiated only when there is a colour map to apply.
        self._fiducial_mask = (
            FiducialDetector(FiducialConfig(dictionary=self.cfg.fiducial_dictionary))
            if self.cfg.marker_color_map else None
        )
        self._sg = (
            StitchGraph(StitchGraphConfig(
                marker_weight=self.cfg.pg_marker_weight,
                odom_weight=self.cfg.pg_odom_weight,
                loop_weight=self.cfg.pg_loop_weight,
                iterations=self.cfg.pg_iterations,
                huber_delta=self.cfg.pg_huber_delta,
            ))
            if self.cfg.use_pose_graph else None
        )
        self._render_idx = 0           # 0-based placement counter == graph rec_idx
        self._cache_dir: Optional[str] = None
        self._finalized = False
        self._last_finalize_report: Optional[dict] = None  # set by finalize()

        # Opaque marker overlay (marker_color_map). Each placed frame's detected
        # marker quads are projected into canvas coords and queued here, then
        # painted solid as the LAST step (see _paint_marker_overlays). Painting
        # after all feather/pyramid blending is what keeps the fill opaque —
        # per-frame recolouring alone gets diluted wherever overlapping frames
        # missed the marker, leaving the see-through pattern.
        self._marker_paints: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
        # rec_idx → mask_markers, kept only when a pose-graph re-render will
        # rebuild the overlay at the globally-corrected poses.
        self._marker_obs: Dict[int, list] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def __del__(self):
        """Shut down the persistent candidate-matching thread pool."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def add_video(
        self,
        video_path: str,
        extract_cfg: Optional[ExtractionConfig] = None,
        verbose: bool = True,
    ) -> None:
        """
        Stream-process an entire video file.
        Never holds more than one decoded frame in memory at a time.

        Periodic GC
        ───────────
        Every gc_interval placed frames, gc.collect() is called and the
        system allocator is asked to return free pages to the OS (via
        malloc_trim on Linux).  This prevents RSS from inflating
        unboundedly due to Python/numpy allocator fragmentation even when
        actual Python-visible memory is flat.
        """
        import gc
        import queue as _queue
        import threading
        import ctypes, sys

        def _trim_allocator():
            """Ask glibc to return free arena pages to the OS (Linux only)."""
            if sys.platform.startswith("linux"):
                try:
                    ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

        gc_interval = 50   # trim every N frames processed (placed or not)
        _DONE = object()   # sentinel: producer signals end-of-stream
        prefetch_q: _queue.Queue = _queue.Queue(maxsize=2)

        def _producer():
            try:
                for frame in stream_frames(video_path, extract_cfg, verbose):
                    # verbose=False: skip messages would interleave with consumer output
                    prefetch_q.put(self._preprocess_frame(frame, verbose=False))
            except Exception as exc:
                prefetch_q.put(exc)
            finally:
                prefetch_q.put(_DONE)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        i = 0
        while True:
            item = prefetch_q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                producer_thread.join()
                raise item
            i += 1
            if item is None:
                # _preprocess_frame skipped this frame (bad keypoints / spread)
                self._n_failed += 1
            else:
                self._place_frame(item, verbose=verbose)
            if verbose and i % 50 == 0:
                print(f"  [frame {i}]  {self.stats}")
            if i % gc_interval == 0:
                gc.collect()
                _trim_allocator()

        producer_thread.join()

        # Stream ended → run the global pose-graph solve and re-render the
        # corrected map. This is the "portion that runs once the feed stops".
        if self.cfg.use_pose_graph:
            self.finalize(verbose=verbose)

    def add_frame(self, img: np.ndarray, verbose: bool = False) -> bool:
        """
        Attempt to place `img` onto the map canvas.
        Returns True if placed, False if alignment failed (frame is skipped).
        """
        prep = self._preprocess_frame(img, verbose=verbose)
        if prep is None:
            self._n_failed += 1
            return False
        return self._place_frame(prep, verbose=verbose)

    def get_map(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        crop: bool = True,
    ) -> np.ndarray:
        """
        Return the current reconstructed map and release the internal canvas.

        Memory note
        ───────────
        After cropping (which produces a view or small copy) the internal
        canvas reference is cleared so the large pre-allocated array can be
        garbage-collected before the resize step.  This means only the
        cropped region + the resized output coexist in memory, rather than
        the full canvas + the output.

        output_shape : (W, H) to resize to; None = natural canvas resolution.
        crop         : trim zero-padding from the content edges first.
        """
        if self._canvas is None:
            raise RuntimeError("No frames placed yet.")

        # If the pose graph is enabled but finalize() hasn't run yet (e.g. the
        # caller used add_frame() directly rather than add_video()), run the
        # global solve + re-render now so get_map() returns the corrected map.
        if self.cfg.use_pose_graph and not self._finalized:
            self.finalize(verbose=False)

        # Paint the solid marker overlay LAST — after all blending / re-render —
        # so the fill stays opaque instead of being feather-diluted to a tint.
        self._paint_marker_overlays()

        canvas = self._canvas
        self._canvas = None   # drop reference; allows GC during resize below

        if crop:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            nz   = cv2.findNonZero(gray)
            del gray
            if nz is not None:
                x, y, cw, ch = cv2.boundingRect(nz)
                canvas = canvas[y : y + ch, x : x + cw].copy()  # contiguous copy
                # The slice above is a view into the full canvas array.
                # The .copy() makes it independent so the canvas can be freed.

        if output_shape is not None:
            result = cv2.resize(canvas, output_shape, interpolation=cv2.INTER_LANCZOS4)
            del canvas   # free cropped region before returning resized result
            return result

        return canvas

    @property
    def stats(self) -> dict:
        ch, cw = self._canvas.shape[:2] if self._canvas is not None else (0, 0)
        return {
            "placed":         self._n_placed,
            "failed":         self._n_failed,
            "keyframes":      len(self._keyframes),
            "canvas_hw":      (ch, cw),
            "canvas_mb":      round(ch * cw * 3 / 1_048_576, 1),
            "grid_refined":   self._n_grid_refined,
            "grid_skipped":   self._n_grid_skipped,
        }

    # ── private helpers ──────────────────────────────────────────────────────

    def _preprocess_frame(
        self, img: np.ndarray, verbose: bool = False
    ) -> Optional[dict]:
        """Preprocessing phase: downscale, feature detection, grid intersections.

        Returns a prep dict consumed by _place_frame, or None if the frame
        should be skipped.  Thread-safe: reads only self.cfg and self.detector,
        both of which are immutable after __init__.
        """
        if self.cfg.processing_scale != 1.0:
            s = self.cfg.processing_scale
            nw = max(1, int(img.shape[1] * s))
            nh = max(1, int(img.shape[0] * s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # img_stitch has masked colours replaced (e.g. yellow → white) and is
        # used only for warping onto the canvas.  Feature extraction always
        # operates on the original unmasked img so no scene content is hidden.
        img_stitch = _apply_color_masks(img, self.cfg.color_masks)

        # Geometric ArUco recolouring (marker_color_map): fill each listed
        # marker's exact quad with a solid colour on the stitch image only.
        # Detected on the unmasked img so colour replacement above doesn't
        # interfere with marker decoding; corners apply directly to img_stitch
        # since both share the same (possibly downscaled) resolution.
        mask_markers: list = []
        if self._fiducial_mask is not None:
            mask_markers = self._fiducial_mask.detect(img)
            img_stitch = _recolor_aruco_markers(
                img_stitch, mask_markers, self.cfg.marker_color_map
            )

        h, w = img.shape[:2]

        if self._sp_session is not None:
            # SuperPoint extractor: no mask support; full image is used.
            kp_px, kp_norm, des_sp = _sp_extract_feats(
                self._sp_session, img, self._sp_model_h, self._sp_model_w
            )
            kp = kp_px                        # (K, 2) pixel coords
            if self._lg_session is not None:
                des = (kp_norm, des_sp)       # packed tuple for LightGlue
            else:
                des = des_sp[0]               # (K, 256) flat array for ratio_test
            if len(kp_px) < 8:
                if verbose:
                    print(f"  [skip] insufficient SP keypoints (n={len(kp_px)})")
                return None
            if verbose:
                print(f"  [sp] kp={len(kp_px)}  img={img.shape[1]}×{img.shape[0]}")
        else:
            feat_mask = _make_feature_mask(
                img,
                self.cfg.feature_exclude_hsv,
                dilate_px=self.cfg.feature_exclude_dilate_px,
            )
            kp, des = _kp_des(self.detector, img, mask=feat_mask)
            if feat_mask is not None:
                del feat_mask
            if des is None or len(kp) < 8:
                if verbose:
                    print(f"  [skip] insufficient keypoints (n={0 if kp is None else len(kp)})")
                return None

        if self.cfg.min_keypoint_bins > 0:
            ok, reason = _keypoint_spread_ok(
                kp, w, h, bins=8, min_filled=self.cfg.min_keypoint_bins
            )
            if not ok:
                if verbose:
                    print(f"  [skip:spread] {reason}")
                return None

        grid_pts: Optional[np.ndarray] = None
        if self.cfg.use_grid_intersections:
            grid_pts = detect_blue_grid_intersections(
                img,
                h_lo=self.cfg.grid_hsv_h_lo,
                h_hi=self.cfg.grid_hsv_h_hi,
                s_lo=self.cfg.grid_hsv_s_lo,
                v_lo=self.cfg.grid_hsv_v_lo,
            )
            if len(grid_pts) == 0:
                grid_pts = None
            elif verbose:
                print(f"  [grid] {len(grid_pts)} intersections detected")

        # ArUco markers — detected on the unmasked image (same as features) so
        # nothing is hidden. Each is a uniquely-identifiable loop-closure
        # landmark for the pose graph.
        markers = self._fiducial.detect(img) if self._fiducial is not None else []
        if markers and verbose:
            print(f"  [aruco] ids={[m.marker_id for m in markers]}")

        return {
            "img_stitch": img_stitch,
            "kp": kp,
            "des": des,
            "grid_pts": grid_pts,
            "markers": markers,
            "mask_markers": mask_markers,
            "h": h,
            "w": w,
        }

    def _place_frame(self, prep: dict, verbose: bool = False) -> bool:
        """Placement phase: first-frame init, parallel candidate matching, warp+blend.

        Must be called from the main thread — mutates self._canvas, self._recent,
        self._keyframes, and the placement/failure counters.
        """
        img_stitch = prep["img_stitch"]
        kp         = prep["kp"]
        des        = prep["des"]
        grid_pts   = prep["grid_pts"]
        markers    = prep.get("markers", [])
        mask_markers = prep.get("mask_markers", [])
        h, w       = prep["h"], prep["w"]

        # ── first frame: initialise canvas ──────────────────────────────────
        if self._canvas is None:
            m = self.cfg.canvas_margin
            if self.cfg.max_canvas_px > 0:
                cap_px = self.cfg.max_canvas_px
                cx = (cap_px - w) // 2
                cy = (cap_px - h) // 2
                H0 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
                self._canvas = np.zeros((cap_px, cap_px, 3), dtype=np.uint8)
                print(f"  Canvas pre-allocated: {cap_px}×{cap_px} px  "
                      f"({cap_px*cap_px*3/1_048_576:.0f} MB)  "
                      f"first frame offset=({cx},{cy})")
            else:
                H0 = np.array([[1, 0, m], [0, 1, m], [0, 0, 1]], dtype=np.float64)
                self._canvas = np.zeros((h + 2 * m, w + 2 * m, 3), dtype=np.uint8)
            self._warp_and_blend_roi(img_stitch, H0)
            self._accumulate_marker_paints(mask_markers, H0)
            self._n_placed = 1
            # Pose graph: first frame is the FIXED gauge keyframe.
            # Call before _register so the returned kf_id can be stored in the
            # keyframe entry that _register is about to add.
            kf_id_assigned = self._graph_on_placed(
                img_stitch, H0, True, w, h, markers, fixed=True,
                mask_markers=mask_markers,
            )
            self._register(kp, des, H0, grid_pts=grid_pts, kf_id=kf_id_assigned)
            return True

        # ── match against recent frames + keyframes (parallel) ───────────────
        lookback = min(self.cfg.lookback, len(self._recent))
        candidates = list(self._recent)[-lookback:] + self._keyframes

        canvas_h_px, canvas_w_px = self._canvas.shape[:2]
        # Allow overflow up to one frame diagonal on each side.
        max_overflow = int(math.hypot(w, h))

        def _match_candidate(ref):
            H_pair, n_in, stats = _pairwise_H(
                (ref["kp"], ref["des"]),
                (kp, des),
                self.norm,
                w, h,
                match_ratio=self.cfg.match_ratio,
                mad_factor=self.cfg.mad_factor,
                grid_pts_ref=ref.get("grid_pts"),
                grid_pts_cur=grid_pts,
                grid_match_dist=self.cfg.grid_match_dist,
                grid_min_matches=self.cfg.grid_min_intersections,
                lg_session=self._lg_session,
            )
            if H_pair is None:
                return None, 0, None, {}, None, None
            composed = ref["H"] @ H_pair
            ok, reason = _validate_composed_H(
                composed, w, h, canvas_h_px, canvas_w_px, max_overflow
            )
            if not ok:
                return None, n_in, reason, {}, None, None
            # Return the reference's kf_id (None for recent-buffer entries) and
            # the raw pairwise H so the caller can add a loop edge to the graph.
            return composed, n_in, None, stats, ref.get("kf_id"), H_pair

        best_H, best_n, best_stats = None, 0, {}
        winner_kf_id: Optional[int] = None
        winner_H_pair: Optional[np.ndarray] = None
        for composed, n_in, reject_reason, stats, ref_kf_id, H_pair in self._pool.map(
            _match_candidate, candidates
        ):
            if composed is None:
                if verbose and reject_reason is not None:
                    print(f"  [reject_composed] {reject_reason} (n_in={n_in})")
                continue
            if n_in > best_n:
                best_H = composed
                best_n = n_in
                best_stats = stats
                winner_kf_id = ref_kf_id
                winner_H_pair = H_pair

        if verbose and best_stats:
            s = best_stats
            print(
                f"  [diag] raw={s['n_raw']} mad={s['n_mad']} "
                f"s1={s['n_in1']} grid={s['n_grid']} s2={s['n_in2']}"
            )

        if best_H is None or best_n < self.cfg.min_inliers:
            self._n_failed += 1
            if verbose:
                print(
                    f"  [skip] #{self._n_placed + self._n_failed}: "
                    f"no reliable H (best_n={best_n})"
                )
            return False

        # ── grow canvas if frame falls outside ──────────────────────────────
        result_H = self._expand_canvas(img_stitch, best_H)
        if result_H is None:
            # _expand_canvas rejected the frame (degenerate overflow).
            self._n_failed += 1
            return False
        best_H = result_H

        # ── warp and blend (ROI only) ────────────────────────────────────────
        self._warp_and_blend_roi(img_stitch, best_H)
        self._accumulate_marker_paints(mask_markers, best_H)
        self._n_placed += 1

        # Feed the pose graph BEFORE _register so the returned kf_id can be
        # stored in the keyframe entry _register is about to create.  Loop
        # edges are only added when the winner was a keyframe (winner_kf_id is
        # not None) and this frame also becomes a keyframe.
        becomes_kf = (self._n_placed % self.cfg.keyframe_interval == 0)
        kf_id_assigned = self._graph_on_placed(
            img_stitch, best_H, becomes_kf, w, h, markers,
            matched_kf_id=winner_kf_id,
            matched_H_pair=winner_H_pair,
            matched_n_inliers=best_n,
            mask_markers=mask_markers,
        )
        self._register(kp, des, best_H, grid_pts=grid_pts, kf_id=kf_id_assigned)
        return True

    # ── pose-graph / fiducial hooks ───────────────────────────────────────────

    def _graph_on_placed(
        self,
        img_stitch: np.ndarray,
        H: np.ndarray,
        is_keyframe: bool,
        w: int,
        h: int,
        markers,
        fixed: bool = False,
        matched_kf_id: Optional[int] = None,
        matched_H_pair: Optional[np.ndarray] = None,
        matched_n_inliers: int = 0,
        mask_markers=(),
    ) -> Optional[int]:
        """Feed a just-placed frame to the pose graph and cache its pixels for
        the finalize() re-render.

        Returns the StitchGraph keyframe id assigned to this frame if it became
        a keyframe (so _register can tag the keyframe entry for future loop
        edges), else None.  Returns None when the pose graph is disabled.

        matched_kf_id / matched_H_pair / matched_n_inliers
            When the winning reference frame during candidate matching was a
            keyframe, these carry its graph id, the pairwise relative transform,
            and the inlier count.  StitchGraph uses them to add a loop-closure
            edge between the two keyframes — the edge that closes the gap
            between lawnmower passes and is the primary non-fiducial constraint.
        """
        if self._sg is None:
            return None
        kf_id = self._sg.on_placed(
            H,
            is_keyframe=is_keyframe,
            frame_w=w,
            frame_h=h,
            fixed=fixed,
            matched_kf_id=matched_kf_id,
            matched_H_pair=matched_H_pair,
            matched_n_inliers=matched_n_inliers,
        )
        self._sg.on_markers(markers)
        self._cache_frame(self._render_idx, img_stitch)
        # Keep this frame's recolour-marker detections so finalize() can rebuild
        # the opaque overlay at the globally-corrected pose (the online quads in
        # self._marker_paints are in pre-correction coords and get discarded).
        if self.cfg.marker_color_map and len(mask_markers):
            self._marker_obs[self._render_idx] = mask_markers
        self._render_idx += 1
        return kf_id

    def _cache_frame(self, idx: int, img: np.ndarray) -> None:
        if self._cache_dir is None:
            if self.cfg.render_cache_dir:
                os.makedirs(self.cfg.render_cache_dir, exist_ok=True)
                self._cache_dir = self.cfg.render_cache_dir
            else:
                import tempfile
                self._cache_dir = tempfile.mkdtemp(prefix="arena_render_")
        cv2.imwrite(os.path.join(self._cache_dir, f"{idx:06d}.png"), img)

    def _load_cached_frame(self, idx: int) -> Optional[np.ndarray]:
        p = os.path.join(self._cache_dir, f"{idx:06d}.png")
        return cv2.imread(p) if os.path.exists(p) else None

    def _cleanup_cache(self) -> None:
        if self.cfg.render_cache_dir:
            return  # caller-owned dir: leave it
        import shutil
        if self._cache_dir and os.path.isdir(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        self._cache_dir = None

    def _accumulate_marker_paints(self, mask_markers, H: np.ndarray) -> None:
        """Project each recolour-listed marker's quad into canvas coords via the
        frame's placement homography H and queue it for the final opaque overlay.

        Queuing (rather than painting now) is deliberate: the paint must land
        AFTER every frame has been blended, otherwise a later overlapping frame
        feather-blends over it and the dilution returns.
        """
        cmap = self.cfg.marker_color_map
        if not cmap or not mask_markers:
            return
        for m in mask_markers:
            bgr = cmap.get(m.marker_id)
            if bgr is None:
                continue
            pts = m.corners_px.reshape(-1, 1, 2).astype(np.float64)
            quad = np.round(cv2.perspectiveTransform(pts, H)).astype(np.int32)
            self._marker_paints.append((quad, bgr))

    def _paint_marker_overlays(self) -> None:
        """Fill every queued marker quad opaquely onto the canvas.

        Called as the LAST step of map generation (after all blending and any
        pose-graph re-render) so the solid colour cannot be diluted. Quads that
        fall partly outside the canvas are clipped by fillConvexPoly.
        """
        if self._canvas is None or not self._marker_paints:
            return
        for quad, bgr in self._marker_paints:
            cv2.fillConvexPoly(self._canvas, quad, bgr)

    def finalize(self, verbose: bool = True) -> Optional[dict]:
        """Run the global pose-graph solve and re-render the corrected map.

        Call this once the stream/video has ended (add_video() calls it
        automatically). The streaming canvas is replaced in-place by a fresh
        render in which every frame is placed at its globally-consistent pose.
        Returns the solver report, or None if the pose graph is disabled.
        """
        if self._sg is None or self._finalized or self._render_idx == 0:
            return None
        if verbose:
            print("  [finalize] solving global pose graph...")
        report = self._sg.finalize(verbose=verbose)

        # Re-render: fresh canvas, re-warp each cached frame at its corrected
        # pose. Frames are pulled from disk one at a time → no RAM blow-up.
        fresh = np.zeros_like(self._canvas)
        old_canvas = self._canvas
        self._canvas = fresh
        del old_canvas
        # Reset the coverage mask so the re-render rebuilds it from scratch.
        self._coverage = np.zeros(fresh.shape[:2], dtype=np.bool_)
        # Discard the online overlay quads (pre-correction coords) and rebuild
        # them below at the corrected poses.
        self._marker_paints = []
        n = 0
        for rec_idx, H_corr in self._sg.iter_corrections():
            img = self._load_cached_frame(rec_idx)
            if img is None:
                continue
            self._warp_and_blend_roi(img, H_corr)
            self._accumulate_marker_paints(self._marker_obs.get(rec_idx, ()), H_corr)
            n += 1
        self._cleanup_cache()
        self._finalized = True
        self._last_finalize_report = report
        if verbose:
            print(f"  [finalize] re-rendered {n} frames | "
                  f"residual {report['rms_before']:.2f} -> {report['rms_after']:.2f} px | "
                  f"markers={report['markers']} edges={report['edges']}")
        return report

    def _register(
        self,
        kp,
        des,
        H: np.ndarray,
        grid_pts: Optional[np.ndarray] = None,
        kf_id: Optional[int] = None,
    ):
        """Store frame data in the recent-frames ring buffer and keyframe archive.

        grid_pts — (N, 2) float32 of detected grid intersection pixel coords
        in this frame's local coordinate system, or None if grid detection
        was disabled / returned no intersections.  Stored alongside kp/des so
        that subsequent frames can use them as Stage-2 refinement anchors.

        kf_id — StitchGraph keyframe id returned by _graph_on_placed when this
        frame became a keyframe.  Stored in the keyframe-archive entry so that
        when a future frame matches this keyframe as its best reference, the
        loop-closure edge can be reported back to the pose graph.
        """
        entry = {
            "kp":       kp,
            "des":      des,
            "H":        H.copy(),
            "grid_pts": grid_pts,   # may be None
            "kf_id":    None,       # recent-buffer entries are not keyframes
        }

        # ── recent ring buffer — deque.maxlen enforces the bound automatically ─
        self._recent.append(entry)

        # ── keyframe archive (bounded by max_keyframes) ──────────────────────
        if self._n_placed % self.cfg.keyframe_interval == 0:
            self._keyframes.append({
                "kp":       kp,
                "des":      des,
                "H":        H.copy(),
                "grid_pts": grid_pts,
                "kf_id":    kf_id,   # links this entry to its pose-graph node
            })

            cap = self.cfg.max_keyframes
            if cap > 0 and len(self._keyframes) > cap:
                # Thin to an evenly-spaced subset so temporal coverage is
                # preserved rather than simply dropping the oldest entries.
                # numpy linspace gives cap indices spread across the list.
                keep = np.linspace(0, len(self._keyframes) - 1, cap, dtype=int)
                self._keyframes = [self._keyframes[i] for i in keep]

    def _expand_canvas(self, img: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """
        Ensure the warped `img` fits within the canvas.

        Returns the (possibly updated) homography, or None if the frame should
        be rejected because its footprint is degenerate or exceeds safety limits.

        Pre-allocated canvas (max_canvas_px > 0)
        ─────────────────────────────────────────
        Near-misses (overflow < canvas_size): warn, clip, accept — the tail of
        the flight path strays slightly outside the pre-allocated area.
        Degenerate overflow (overflow >= canvas_size): reject with [reject_overflow]
        — the composed homography has drifted and the frame contributes nothing.

        Dynamic canvas (max_canvas_px == 0)
        ────────────────────────────────────
        A hard cap of max(canvas_current_size × 4, 20000 px) is enforced before
        any allocation.  Exceeding it means the composed H is degenerate (not
        that the arena is genuinely that large) and the frame is rejected.
        """
        fh, fw = img.shape[:2]
        corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        ch, cw = self._canvas.shape[:2]

        if self.cfg.max_canvas_px > 0:
            # Pre-allocated path
            out_l = float(-wc[:, 0].min())
            out_t = float(-wc[:, 1].min())
            out_r = float(wc[:, 0].max() - cw)
            out_b = float(wc[:, 1].max() - ch)
            worst = max(out_l, out_t, out_r, out_b)

            if worst <= 10:
                return H  # fully inside, fast path

            # Overflow larger than the canvas itself → degenerate composed H.
            # _validate_composed_H should catch this first, but belt-and-suspenders.
            if worst >= max(cw, ch):
                print(
                    f"  [reject_overflow] Footprint overflow={worst:.0f}px "
                    f">= canvas_size={max(cw, ch)}px — degenerate H rejected."
                )
                return None

            # Legitimate slight overshoot (drone near canvas edge): warn and clip.
            print(
                f"  [warn] Frame clips canvas edge "
                f"(l={out_l:.0f} t={out_t:.0f} r={out_r:.0f} b={out_b:.0f} px). "
                f"Consider increasing max_canvas_px."
            )
            return H  # _warp_and_blend_roi clips to canvas bounds

        # Dynamic path: allocate + copy + retranslate.
        pl = max(0, int(-wc[:, 0].min()) + 10)
        pt = max(0, int(-wc[:, 1].min()) + 10)
        pr = max(0, int(wc[:, 0].max()) - cw + 10)
        pb = max(0, int(wc[:, 1].max()) - ch + 10)

        if pl == 0 and pt == 0 and pr == 0 and pb == 0:
            return H  # no expansion needed

        new_w = cw + pl + pr
        new_h = ch + pt + pb

        # Safety cap: if the required canvas is implausibly large it means the
        # composed H is degenerate.  Using max(current_size×4, 20000) as the
        # ceiling is generous for any realistic indoor arena.
        max_dynamic = max(cw * 4, ch * 4, 20_000)
        if new_w > max_dynamic or new_h > max_dynamic:
            print(
                f"  [reject_overflow] Dynamic expansion would reach "
                f"{new_w}×{new_h} px (limit={max_dynamic}px) — degenerate H rejected."
            )
            return None

        new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_canvas[pt : pt + ch, pl : pl + cw] = self._canvas
        del self._canvas        # release old before assigning new
        self._canvas = new_canvas

        # Expand the coverage mask by the same offsets.
        if self._coverage is not None:
            new_cov = np.zeros((new_h, new_w), dtype=np.bool_)
            new_cov[pt : pt + ch, pl : pl + cw] = self._coverage
            self._coverage = new_cov

        T = np.array([[1, 0, pl], [0, 1, pt], [0, 0, 1]], dtype=np.float64)
        for rec in (*self._recent, *self._keyframes):
            rec["H"] = T @ rec["H"]

        # Already-queued marker overlays live in canvas coords → shift them too.
        if self._marker_paints:
            shift = np.array([[[pl, pt]]], dtype=np.int32)
            self._marker_paints = [(q + shift, c) for q, c in self._marker_paints]

        return T @ H

    def _warp_and_blend_roi(self, img: np.ndarray, H: np.ndarray):
        """
        Memory-efficient warp + blend.

        The critical difference from the original implementation:

          Before: cv2.warpPerspective(img, H, (canvas_w, canvas_h))
                  → allocates one full-canvas array per frame (~140 MB for
                    an 8 000×6 000 canvas), regardless of how small the
                    incoming frame is.

          After:  compute the tight bounding box of the warped frame on
                  the canvas, shift H by (-x0, -y0), then:
                  cv2.warpPerspective(img, H_roi, (roi_w, roi_h))
                  → allocates only the frame footprint (~2–4 MB for a
                    1080p frame), regardless of canvas size.

        Peak extra memory ≈ 2 × (roi_h × roi_w × 3) bytes.

        Coverage mask
        ─────────────
        A boolean mask (self._coverage) is maintained alongside the canvas.
        It is set True at every pixel that has been painted, regardless of the
        pixel's colour value.  This correctly handles genuinely black scene
        content (black obstacles, dark floor) that would otherwise be
        misidentified as "canvas not yet painted" by a pixel-sum > 0 test.

        Frame footprint (mask_new)
        ──────────────────────────
        mask_new is derived by warping a white sentinel image through the same
        transform, not from the pixel values of warped_roi.  This marks the
        true source-frame boundary so that genuinely black source pixels within
        the footprint are counted as covered rather than excluded.
        """
        fh, fw = img.shape[:2]
        ch, cw = self._canvas.shape[:2]

        # Lazily initialise the coverage mask (first call after canvas creation).
        if self._coverage is None:
            self._coverage = np.zeros((ch, cw), dtype=np.bool_)

        # ── tight bounding box of the warped frame on the canvas ─────────────
        corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        x0 = max(0, int(math.floor(wc[:, 0].min())))
        y0 = max(0, int(math.floor(wc[:, 1].min())))
        x1 = min(cw, int(math.ceil(wc[:, 0].max())) + 1)
        y1 = min(ch, int(math.ceil(wc[:, 1].max())) + 1)

        if x1 <= x0 or y1 <= y0:
            return  # frame entirely outside canvas (should not happen)

        roi_w, roi_h = x1 - x0, y1 - y0

        # ── sub-homography: shift origin to (x0, y0) ────────────────────────
        T_shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        H_roi = T_shift @ H

        # ── warp only into the ROI — frame-sized, not canvas-sized ───────────
        if _CUDA_AVAILABLE:
            _gpu_src = cv2.cuda_GpuMat()
            _gpu_src.upload(img)
            warped_roi = cv2.cuda.warpPerspective(
                _gpu_src, H_roi.astype(np.float32), (roi_w, roi_h)
            ).download()
        else:
            warped_roi = cv2.warpPerspective(img, H_roi, (roi_w, roi_h))

        # True footprint: warp a white sentinel so that genuinely black source
        # pixels are counted as "inside the frame" rather than "no data".
        # cv2.warpPerspective fills out-of-source-bounds areas with 0, so any
        # non-zero sentinel pixel correctly marks a source-frame location.
        _sentinel = np.ones((fh, fw), dtype=np.uint8) * 255
        mask_new = cv2.warpPerspective(_sentinel, H_roi, (roi_w, roi_h)) > 0
        del _sentinel

        # ── coverage ROI — a VIEW into self._coverage (writes propagate) ─────
        coverage_roi = self._coverage[y0:y1, x0:x1]   # bool (roi_h, roi_w)

        # ── blend directly into the canvas slice ─────────────────────────────
        # canvas_roi is a VIEW into self._canvas, so in-place writes propagate.
        canvas_roi = self._canvas[y0:y1, x0:x1]
        mode = self.cfg.blend_mode

        if mode == "flat":
            mask_c = coverage_roi
            only_new = mask_new & ~mask_c
            overlap = mask_new & mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                canvas_roi[overlap] = (
                    canvas_roi[overlap].astype(np.float32) * 0.5
                    + warped_roi[overlap].astype(np.float32) * 0.5
                ).astype(np.uint8)

        elif mode == "pyramid":
            mask_c = coverage_roi
            only_new = mask_new & ~mask_c
            overlap = mask_new & mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                # Dual distance transform for the alpha map — same semantics as
                # feather mode: seam placed at the iso-depth contour of both
                # footprints.  This also fixes the old single-distanceTransform
                # on the overlap region, which gave the new frame higher weight
                # at the overlap centre rather than at its own territory centre.
                dist_new = cv2.distanceTransform(
                    mask_new.astype(np.uint8) * 255, cv2.DIST_L2, 5
                )
                dist_old = cv2.distanceTransform(
                    mask_c.astype(np.uint8) * 255, cv2.DIST_L2, 5
                )
                alpha = (dist_new / (dist_new + dist_old + 1e-6)).astype(np.float32)
                blended = _laplacian_pyramid_blend(
                    canvas_roi, warped_roi, alpha, self.cfg.pyramid_levels
                )
                canvas_roi[overlap] = blended[overlap]

        else:  # "feather" (default)
            blended = _feather_blend_roi(canvas_roi, warped_roi, mask_new,
                                         mask_c=coverage_roi)
            self._canvas[y0:y1, x0:x1] = blended

        # ── update coverage for every pixel touched by this frame ─────────────
        coverage_roi |= mask_new


# ══════════════════════════════════════════════════════════════════════════════
# Convenience top-level function
# ══════════════════════════════════════════════════════════════════════════════


def reconstruct_from_video(
    video_path: str,
    output_shape: Optional[Tuple[int, int]] = None,
    extract_cfg: Optional[ExtractionConfig] = None,
    reconstruct_cfg: Optional[ReconstructConfig] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Full pipeline: .mp4 drone video → stitched top-down map image.

    Memory profile (with default max_canvas_px=8000)
    ─────────────────────────────────────────────────
    Fixed cost (allocated once, freed when this function returns):
      canvas          ~183 MB   (8000×8000×3, pre-allocated)
      _recent buffer   ~20 MB   (8 frames × SIFT descriptors)
      _keyframes       ~50 MB   (max 20 keyframes × SIFT descriptors)

    Per-frame transient (freed before next frame):
      frame BGR        ~6 MB
      img_stitch copy  ~6 MB    (color-masked copy)
      img_hsv          ~6 MB    (freed inside _apply_color_masks)
      gray             ~2 MB    (freed after keypoint detection)
      Laplacian f32    ~8 MB    (freed inside _blur_score)
      warped ROI       ~6 MB    (frame-footprint warp only)
      Total transient ~34 MB

    Peak: canvas + _recent + _keyframes + transient ≈ 287 MB

    No canvas-expansion spikes (dynamic reallocation eliminated).
    """
    _sep = "─" * 60

    if verbose:
        print(_sep)
        print(f"  Streaming reconstruction  ←  {video_path!r}")
        print(_sep)

    rec = MapReconstructor(reconstruct_cfg)
    rec.add_video(video_path, extract_cfg=extract_cfg, verbose=verbose)

    if verbose:
        print()
        print(_sep)
        print(f"  Finalising map  |  {rec.stats}")
        print(_sep)

    result = rec.get_map(output_shape=output_shape)

    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(save_path, result)
        if not ok:
            raise IOError(
                f"cv2.imwrite failed for path: {save_path!r}\n"
                f"  Directory exists: {os.path.isdir(out_dir)}\n"
                f"  Check the path is writable and the extension is supported."
            )
        if verbose:
            print(f"  Saved → {save_path}")

    return result