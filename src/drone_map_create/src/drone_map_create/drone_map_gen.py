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

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Shared low-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_detector():
    """SIFT with generous feature budget; ORB fallback for older OpenCV builds."""
    try:
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    except AttributeError:
        return cv2.ORB_create(nfeatures=8000), cv2.NORM_HAMMING


def _kp_des(detector, img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return detector.detectAndCompute(gray, None)


def _match(des1, des2, norm, ratio: float = 0.80) -> list:
    """Lowe ratio-test match between two descriptor sets."""
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(des1, des2, k=2)
    return [
        m
        for pair in raw
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    ]

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

def _pairwise_H(
    feat_ref: tuple,
    feat_cur: tuple,
    norm: int,
    frame_w: int,
    frame_h: int,
) -> tuple:
    """
    Compute H mapping cur → ref coordinate space via RANSAC.
    Returns (H, n_inliers) or (None, 0) if the homography is unreliable.
    """
    kp_r, des_r = feat_ref
    kp_c, des_c = feat_cur
    if des_r is None or des_c is None or len(kp_r) < 8 or len(kp_c) < 8:
        return None, 0

    good = _match(des_r, des_c, norm, ratio=0.80)
    if len(good) < 8:
        return None, 0

    pts_r = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_c,
        pts_r,
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
        confidence=0.995,
    )
    if H is None:
        return None, 0

    n_in = int(mask.sum()) if mask is not None else 0
    if n_in < 8:
        return None, 0

    ok, reason = _validate_homography(H, frame_w, frame_h)
    if not ok:
        return None, 0

    cx, cy = frame_w / 2.0, frame_h / 2.0
    mapped = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0][0]
    if abs(mapped[0]) > frame_w * 6 or abs(mapped[1]) > frame_h * 6:
        return None, 0

    return H, n_in


# ══════════════════════════════════════════════════════════════════════════════
# Frame quality assessment
# ══════════════════════════════════════════════════════════════════════════════


def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance – higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _codec_artifact_ratio(gray: np.ndarray) -> float:
    """
    Estimate DCT 8×8 block-artifact severity.
    Compares mean absolute differences at 8-pixel-spaced column boundaries
    to the overall mean column-difference.
    """
    all_diff = np.abs(np.diff(gray.astype(np.int16), axis=1))
    mean_all = float(all_diff.mean()) + 1e-6
    boundary_cols = np.arange(7, gray.shape[1] - 1, 8)
    if boundary_cols.size == 0:
        return 1.0
    return float(all_diff[:, boundary_cols].mean() / mean_all)


def _assess_frame(
    img: np.ndarray,
    blur_thresh: float,
    artifact_thresh: float,
    lo_brightness: float = 15.0,
    hi_brightness: float = 240.0,
) -> Tuple[bool, str]:
    """Return (ok, reason_string). reason is 'ok' when the frame passes."""
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
    prev_kp = prev_des = None
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

            # ── quality gate ───────────────────────────────────────────────
            ok, reason = _assess_frame(img, cfg.blur_thresh, cfg.artifact_thresh)
            if not ok:
                stats["quality"] += 1
                if verbose:
                    print(f"  [drop:quality]  frame {frame_idx:5d}: {reason}")
                continue

            # ── movement gate ──────────────────────────────────────────────
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = detector.detectAndCompute(gray, None)

            if prev_des is not None and des is not None and len(kp) >= 8:
                good = _match(prev_des, des, norm, ratio=0.75)
                if len(good) >= 8:
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good])
                    diag = math.hypot(img.shape[1], img.shape[0])
                    mv = float(np.linalg.norm(pts1 - pts2, axis=1).mean()) / diag

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

            prev_kp, prev_des = kp, des
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


# ══════════════════════════════════════════════════════════════════════════════
# Blending utilities
# ══════════════════════════════════════════════════════════════════════════════


def _feather_blend_roi(
    canvas_roi: np.ndarray,
    warped_roi: np.ndarray,
    mask_new: np.ndarray,
) -> np.ndarray:
    """
    Distance-weighted feathering blend operating on pre-extracted ROI arrays.
    Both inputs are the same small region; the full canvas is never touched.
    """
    mask_c = canvas_roi.sum(axis=2) > 0
    only_new = mask_new & ~mask_c
    overlap = mask_new & mask_c

    result = canvas_roi.copy()
    result[only_new] = warped_roi[only_new]

    if not overlap.any():
        return result

    dist = cv2.distanceTransform(overlap.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    d_max = dist.max()
    alpha = (dist / d_max if d_max > 0 else np.full_like(dist, 0.5)).astype(np.float32)
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
    for la, lb, gm in zip(lp_a, lp_b, reversed(gp_m)):
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
    replace_bgr: Tuple[int, int, int] = (255, 255, 255)   # white

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


def _apply_color_masks(img: np.ndarray, masks: List[ColorRangeMask]) -> np.ndarray:
    """
    Replace pixels matching any of the given HSV color ranges with each
    mask's replacement colour (default: white).

    The function converts `img` to HSV once and evaluates all masks in a
    single pass.  Returns a *copy* of `img` with the matching pixels
    recoloured; the original array is never modified.

    This is called on each frame **before** warping onto the canvas so that
    the stitched map never contains the masked colours.  Feature detection
    for homography estimation always runs on the *original* unmasked frame
    so that keypoints are not degraded by the solid replacement colour.
    """
    if not masks:
        return img

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result  = img.copy()

    for cm in masks:
        lo   = np.array([cm.h_lo, cm.s_lo, cm.v_lo], dtype=np.uint8)
        hi   = np.array([cm.h_hi, cm.s_hi, cm.v_hi], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lo, hi)
        if mask.any():
            result[mask > 0] = cm.replace_bgr

    return result


@dataclass
class ReconstructConfig:
    """Tunable parameters for incremental frame stitching."""

    canvas_margin: int = 2000
    """Initial blank padding (px) around the first frame."""

    min_inliers: int = 10
    """Minimum RANSAC inliers to accept a homography."""

    lookback: int = 4
    """Number of recently placed frames to try matching against."""

    keyframe_interval: int = 15
    """Cache a long-range keyframe every N successfully placed frames."""

    blend_mode: str = "feather"
    """
    "feather"  – distance-weighted linear blend  (fast, default)
    "pyramid"  – Laplacian pyramid multi-band     (best quality, slower)
    "flat"     – simple 50/50 average             (debug / benchmark)
    """

    pyramid_levels: int = 4
    """Laplacian pyramid depth; only used when blend_mode='pyramid'."""

    processing_scale: float = 1.0
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
        self.detector, self.norm = _make_detector()

        self._canvas: Optional[np.ndarray] = None
        self._recent: List[dict] = []  # ring buffer  {kp, des, H}
        self._keyframes: List[dict] = []  # sparse long-range anchors

        self._n_placed = 0
        self._n_failed = 0

    # ── public API ───────────────────────────────────────────────────────────

    def add_video(
        self,
        video_path: str,
        extract_cfg: Optional[ExtractionConfig] = None,
        verbose: bool = True,
    ) -> None:
        """
        Stream-process an entire video file.
        Never holds more than one decoded frame in memory at a time.
        """
        for i, frame in enumerate(stream_frames(video_path, extract_cfg, verbose), 1):
            self.add_frame(frame, verbose=verbose)
            if verbose and i % 50 == 0:
                print(f"  [frame {i}]  {self.stats}")

    def add_frame(self, img: np.ndarray, verbose: bool = False) -> bool:
        """
        Attempt to place `img` onto the map canvas.
        Returns True if placed, False if alignment failed (frame is skipped).
        """
        # ── optional downscale ───────────────────────────────────────────────
        if self.cfg.processing_scale != 1.0:
            s = self.cfg.processing_scale
            nw = max(1, int(img.shape[1] * s))
            nh = max(1, int(img.shape[0] * s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # ── colour masking (pre-stitch) ──────────────────────────────────────
        # img_stitch has the masked colours replaced (e.g. yellow → white).
        # img is kept unmasked so that keypoint detection uses the full
        # original appearance, preserving homography quality.
        img_stitch = _apply_color_masks(img, self.cfg.color_masks)

        h, w = img.shape[:2]
        kp, des = _kp_des(self.detector, img)
        if des is None or len(kp) < 8:
            self._n_failed += 1
            return False

        # ── first frame: initialise canvas ──────────────────────────────────
        if self._canvas is None:
            m = self.cfg.canvas_margin
            H0 = np.array([[1, 0, m], [0, 1, m], [0, 0, 1]], dtype=np.float64)
            self._canvas = np.zeros((h + 2 * m, w + 2 * m, 3), dtype=np.uint8)
            self._warp_and_blend_roi(img_stitch, H0)
            self._register(kp, des, H0)
            self._n_placed = 1
            return True

        # ── match against recent frames + keyframes ──────────────────────────
        lookback = min(self.cfg.lookback, len(self._recent))
        candidates = self._recent[-lookback:] + self._keyframes

        best_H, best_n = None, 0
        for ref in candidates:
            H_pair, n_in = _pairwise_H(
                (ref["kp"], ref["des"]),
                (kp, des),
                self.norm,
                w,
                h,
            )
            if H_pair is not None and n_in > best_n:
                best_H = ref["H"] @ H_pair
                best_n = n_in

        if best_H is None or best_n < self.cfg.min_inliers:
            self._n_failed += 1
            if verbose:
                print(
                    f"  [skip] #{self._n_placed + self._n_failed}: "
                    f"no reliable H (best_n={best_n})"
                )
            return False

        # ── grow canvas if frame falls outside ──────────────────────────────
        best_H = self._expand_canvas(img, best_H)

        # ── warp and blend (ROI only — the key memory fix) ──────────────────
        self._warp_and_blend_roi(img_stitch, best_H)
        self._n_placed += 1
        self._register(kp, des, best_H)
        return True

    def get_map(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        crop: bool = True,
    ) -> np.ndarray:
        """
        Return the current reconstructed map.

        output_shape : (W, H) to resize to; None = natural canvas resolution.
        crop         : trim zero-padding from the content edges first.
        """
        if self._canvas is None:
            raise RuntimeError("No frames placed yet.")

        result = self._canvas
        if crop:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            nz = cv2.findNonZero(gray)
            if nz is not None:
                x, y, cw, ch = cv2.boundingRect(nz)
                result = result[y : y + ch, x : x + cw]

        if output_shape is not None:
            result = cv2.resize(result, output_shape, interpolation=cv2.INTER_LANCZOS4)

        return result

    @property
    def stats(self) -> dict:
        ch, cw = self._canvas.shape[:2] if self._canvas is not None else (0, 0)
        return {
            "placed": self._n_placed,
            "failed": self._n_failed,
            "keyframes": len(self._keyframes),
            "canvas_hw": (ch, cw),
            "canvas_mb": round(ch * cw * 3 / 1_048_576, 1),
        }

    # ── private helpers ──────────────────────────────────────────────────────

    def _register(self, kp, des, H: np.ndarray):
        entry = {"kp": kp, "des": des, "H": H.copy()}
        self._recent.append(entry)
        if len(self._recent) > self.cfg.lookback + 2:
            self._recent.pop(0)
        if self._n_placed % self.cfg.keyframe_interval == 0:
            self._keyframes.append({"kp": kp, "des": des, "H": H.copy()})

    def _expand_canvas(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Pad the canvas so the warped `img` fits, translating all stored
        homographies accordingly.  Returns the updated H for `img`.
        """
        fh, fw = img.shape[:2]
        corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        ch, cw = self._canvas.shape[:2]
        pl = max(0, int(-wc[:, 0].min()) + 10)
        pt = max(0, int(-wc[:, 1].min()) + 10)
        pr = max(0, int(wc[:, 0].max()) - cw + 10)
        pb = max(0, int(wc[:, 1].max()) - ch + 10)

        if pl == 0 and pt == 0 and pr == 0 and pb == 0:
            return H

        new_canvas = np.zeros((ch + pt + pb, cw + pl + pr, 3), dtype=np.uint8)
        new_canvas[pt : pt + ch, pl : pl + cw] = self._canvas
        self._canvas = new_canvas

        T = np.array([[1, 0, pl], [0, 1, pt], [0, 0, 1]], dtype=np.float64)
        for rec in (*self._recent, *self._keyframes):
            rec["H"] = T @ rec["H"]

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
        """
        fh, fw = img.shape[:2]
        ch, cw = self._canvas.shape[:2]

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
        warped_roi = cv2.warpPerspective(img, H_roi, (roi_w, roi_h))
        mask_new = warped_roi.sum(axis=2) > 0

        # ── blend directly into the canvas slice ─────────────────────────────
        # canvas_roi is a VIEW into self._canvas, so in-place writes propagate.
        canvas_roi = self._canvas[y0:y1, x0:x1]
        mode = self.cfg.blend_mode

        if mode == "flat":
            mask_c = canvas_roi.sum(axis=2) > 0
            only_new = mask_new & ~mask_c
            overlap = mask_new & mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                canvas_roi[overlap] = (
                    canvas_roi[overlap].astype(np.float32) * 0.5
                    + warped_roi[overlap].astype(np.float32) * 0.5
                ).astype(np.uint8)

        elif mode == "pyramid":
            mask_c = canvas_roi.sum(axis=2) > 0
            only_new = mask_new & ~mask_c
            overlap = mask_new & mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                dist = cv2.distanceTransform(
                    overlap.astype(np.uint8) * 255, cv2.DIST_L2, 5
                )
                d_max = dist.max()
                alpha = (dist / d_max if d_max > 0 else np.full_like(dist, 0.5)).astype(
                    np.float32
                )
                blended = _laplacian_pyramid_blend(
                    canvas_roi, warped_roi, alpha, self.cfg.pyramid_levels
                )
                canvas_roi[overlap] = blended[overlap]

        else:  # "feather" (default)
            blended = _feather_blend_roi(canvas_roi, warped_roi, mask_new)
            self._canvas[y0:y1, x0:x1] = blended


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

    Frames are streamed one at a time; total RAM is approximately:
        canvas_size  +  2 × one_frame_size
    regardless of video length.

    Memory-saving knobs
    -------------------
    • Lower target_fps (fewer frames → slower canvas growth)
    • Set processing_scale=0.5 (4× smaller canvas and per-frame buffers)
    • Both together:
        reconstruct_from_video(
            "flight.mp4",
            extract_cfg=ExtractionConfig(target_fps=3.0),
            reconstruct_cfg=ReconstructConfig(processing_scale=0.5),
        )
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
        cv2.imwrite(save_path, result)
        if verbose:
            print(f"  Saved → {save_path}")

    return result