"""
drone_map_csv_gen.py
─────────────────────────────────────────────────────────────────────────────
CSV-driven drone map reconstruction.

What this module replaces
─────────────────────────
The previous module (drone_map_gen.py) used SIFT + RANSAC homography chaining
to align frames against each other.  Every frame's placement was relative to
its predecessor, so small errors compounded into the fan/ray artifacts.

This module takes per-frame poses from an OptiTrack CSV instead:

    timestamp_sec, frame_id, pos_x, pos_y, pos_z, yaw

Position (pos_x, pos_y, pos_z) is from motion capture and trusted absolutely.
yaw is from the drone IMU and known to be unreliable, so this module offers
two opt-in correction stages:

  1. Global yaw-offset calibration   (calibrate_yaw_offset=True)
     Runs per-frame refinement on the first N frames and applies the median
     correction as a single constant offset to every CSV yaw thereafter.
     Catches a constant IMU-to-world misalignment (the common case).

  2. Per-frame yaw refinement        (refine_yaw=True)
     For each frame, after placing at CSV(x, y, z) + corrected yaw, match
     features against the existing canvas content under the frame footprint
     and solve a closed-form bounded rotation.  Translation and scale stay
     locked to the OptiTrack value.  If the rotation exceeds the threshold
     or matching fails, fall back silently to CSV.

Both flags are independent — disabling both is "trust the CSV completely",
which is a useful sanity check first run.

Retained from drone_map_gen.py
──────────────────────────────
  • frame quality assessment (blur, codec artifacts, brightness)
  • HSV colour replacement before stitching  (ColorRangeMask, color_masks)
  • HSV feature-detection exclusion mask     (now used only by yaw refinement)
  • Laplacian pyramid blending               (and feather/flat alternatives)
  • ROI warping (memory-efficient — frame-sized scratch, not canvas-sized)
  • Symmetric codec artifact detection
  • Pre-allocated canvas (no expansion spikes)
  • SIFT detector + mutual-cross-check matcher

Dropped
───────
  • _pairwise_H + RANSAC homography chain        (replaced by CSV pose)
  • Keyframes, lookback ring buffer              (every frame independent)
  • _validate_composed_H / drift checks          (no chain to drift)
  • Dynamic canvas expansion                     (size known a priori)
  • SIFT-based movement gate                     (replaced by CSV delta)

Camera intrinsics
─────────────────
You MUST configure CameraIntrinsics for your camera or the absolute scale
is wrong.  The two ways:

    CameraIntrinsics(fx=1370.0, fy=1370.0, cx=960.0, cy=540.0)

or, if you only know FOV:

    CameraIntrinsics.from_fov(frame_w=1920, frame_h=1080, h_fov_deg=70.0)

Sign conventions
────────────────
Three sign flags (in CoordinateConfig) handle camera mounting variation.
First run, look at the output: if it's mirrored, flip `image_x_sign`; if
upside-down, flip `image_y_sign`; if rotation grows in the wrong direction
between frames, flip `yaw_sign`.

Quick start
───────────
    from drone_map_csv_gen import (
        reconstruct_from_csv, CSVStitchConfig, CameraIntrinsics,
        CoordinateConfig, ColorRangeMask,
    )

    cfg = CSVStitchConfig(
        intrinsics = CameraIntrinsics.from_fov(1920, 1080, h_fov_deg=70.0),
        coord_cfg  = CoordinateConfig(),  # defaults; flip signs if needed
        color_masks=[ColorRangeMask.yellow(), ColorRangeMask.brown()],
        feature_exclude_hsv=[ColorRangeMask.blue_tape()],
        blend_mode="pyramid",
        refine_yaw=True,
        calibrate_yaw_offset=True,
    )

    result = reconstruct_from_csv(
        video_path="flight.mp4",
        csv_path="poses.csv",
        cfg=cfg,
        save_path="map.png",
    )
"""

from __future__ import annotations

import csv
import math
import os
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Re-use what still applies from the original module.
# All of these are still correct for the CSV-driven flow.
from drone_map_create.drone_map_gen import (
    ColorRangeMask,
    _apply_color_masks,
    _assess_frame,
    _feather_blend_roi,
    _kp_des,
    _laplacian_pyramid_blend,
    _make_detector,
    _make_feature_mask,
    _match,
)


# ══════════════════════════════════════════════════════════════════════════════
# Camera intrinsics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels.

    fx, fy : focal lengths (pixels).  Almost always close to equal.
    cx, cy : principal point (pixels).  Usually ≈ (image_w/2, image_h/2).

    For a downward-looking camera at altitude z, one image pixel covers
    z/fx metres of ground horizontally and z/fy vertically.  This is the
    only place the absolute world-to-pixel scale enters the pipeline —
    get it wrong and the stitched map will be the wrong physical size
    (everything else still works, but distances are off).

    dist_coeffs
    ───────────
    Optional OpenCV-format distortion coefficients from cv2.calibrateCamera:
        (k1, k2, p1, p2)            — minimum 4 (radial + tangential)
        (k1, k2, p1, p2, k3)        — standard 5
        (k1, k2, p1, p2, k3, k4, k5, k6)  — rational model

    Providing this field is the declaration that the intrinsics come from a
    real calibration rather than a guessed FOV.  It is consumed only when
    CSVStitchConfig.undistort_frames=True; otherwise frames are used as-is
    and the field is informational.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[Tuple[float, ...]] = None

    @classmethod
    def from_fov(
        cls,
        frame_w: int,
        frame_h: int,
        h_fov_deg: float,
        v_fov_deg: Optional[float] = None,
    ) -> "CameraIntrinsics":
        """Build intrinsics from horizontal (and optional vertical) FOV.

        If v_fov_deg is omitted, fy is set equal to fx (square pixels — true
        for almost all modern sensors).  Principal point is assumed at the
        image centre; override with the standard constructor if you know
        better from a calibration target.
        """
        fx = frame_w / (2.0 * math.tan(math.radians(h_fov_deg) / 2.0))
        if v_fov_deg is None:
            fy = fx
        else:
            fy = frame_h / (2.0 * math.tan(math.radians(v_fov_deg) / 2.0))
        return cls(fx=fx, fy=fy, cx=frame_w / 2.0, cy=frame_h / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Coordinate sign / orientation conventions
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoordinateConfig:
    """Sign conventions for the image ↔ world ↔ canvas mapping.

    These depend on how the camera is mounted on the drone and on what
    "yaw" means in your motion-capture frame.  Defaults are the most common
    case (camera looks down, image-up = world-north at yaw=0, yaw is CCW),
    but you may need to flip one or more signs after looking at the output.

    Calibration recipe
    ──────────────────
    1. Run with defaults, no yaw refinement.
    2. Look at the produced map alongside the first few frames:
         • Mirrored left-right       → flip image_x_sign
         • Upside down               → flip image_y_sign
         • Each new frame appears
           rotated wrong relative
           to its CSV yaw            → flip yaw_sign
    3. Re-run.  Repeat at most twice.

    yaw_offset_rad
    ──────────────
    Set manually only if you've measured it.  Normally leave at 0 and let
    calibrate_yaw_offset=True estimate it from the data.
    """

    image_x_sign: int = +1
    """+1 if image-right corresponds to world-east when yaw=0; -1 otherwise."""

    image_y_sign: int = +1
    """+1 if image-down corresponds to world-south when yaw=0 (i.e. the
    image y-axis aligns with the negative world y-axis at yaw=0).  -1 if
    your camera's y is mounted the opposite way."""

    yaw_sign: int = +1
    """+1 if yaw increases counter-clockwise looking down (standard math
    convention); -1 if your CSV yaw is clockwise."""

    yaw_offset_rad: float = 0.0
    """Constant offset added to (yaw_sign * yaw_csv) before building the warp.
    Leave 0 and let yaw-offset calibration estimate this if you don't know it."""

    canvas_y_flip: bool = True
    """If True, canvas y increases downward (image convention) while world y
    increases upward (north).  Almost always correct; set False only if you
    want a "world-aligned" canvas where +y is up."""


# ══════════════════════════════════════════════════════════════════════════════
# CSV pose loading and lookup
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Pose:
    """One row from the pose CSV."""
    frame_id: int
    t_sec:    float
    x:        float
    y:        float
    z:        float
    yaw:      float


class PoseTable:
    """Loads pose CSV and supports both frame-id and timestamp lookup.

    The CSV is loaded once into RAM (a few hundred KB even for long flights).
    Lookups are O(1) for frame_id (hash table) and O(log N) for timestamp
    (binary search over a sorted list).

    Frame index → CSV row
    ─────────────────────
    Video frame indexing is 0-based (OpenCV) but CSV frame_id usually starts
    at 1.  `frame_id_offset` (default 1) bridges them: video frame i is
    looked up as CSV frame_id (i + frame_id_offset).

    If your CSV is at a different rate than the video (e.g. OptiTrack
    streaming at 120 Hz while the video is at 30 Hz), prefer timestamp
    lookup: pass the video frame's timestamp (frame_idx / video_fps) to
    `by_timestamp()`.
    """

    def __init__(self, csv_path: str, frame_id_offset: int = 1):
        self.frame_id_offset = frame_id_offset
        self._by_fid: dict = {}
        self._timestamps: List[float] = []
        self._poses_by_ts: List[Pose] = []
        self._load(csv_path)

    def _load(self, path: str) -> None:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"frame_id", "timestamp_sec", "pos_x", "pos_y", "pos_z", "yaw"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSV {path!r} is missing required columns: {sorted(missing)}"
                )
            rows: List[Pose] = []
            for r in reader:
                p = Pose(
                    frame_id=int(r["frame_id"]),
                    t_sec=float(r["timestamp_sec"]),
                    x=float(r["pos_x"]),
                    y=float(r["pos_y"]),
                    z=float(r["pos_z"]),
                    yaw=float(r["yaw"]),
                )
                rows.append(p)
                self._by_fid[p.frame_id] = p

        # Sort by timestamp for binary-search lookup.
        rows.sort(key=lambda p: p.t_sec)
        self._poses_by_ts = rows
        self._timestamps = [p.t_sec for p in rows]

    def by_frame_id(self, video_frame_idx: int) -> Optional[Pose]:
        """Look up by video frame index (0-based).  Adds frame_id_offset
        internally so a 0-based video index becomes the matching CSV frame_id."""
        return self._by_fid.get(video_frame_idx + self.frame_id_offset)

    def by_timestamp(self, t_sec: float, max_dt_sec: float = 0.05) -> Optional[Pose]:
        """Nearest-timestamp lookup.  Returns None if the closest row is more
        than `max_dt_sec` away (safer than silently picking a far-off pose)."""
        if not self._timestamps:
            return None
        i = bisect_left(self._timestamps, t_sec)
        cands = []
        if i > 0:
            cands.append(i - 1)
        if i < len(self._timestamps):
            cands.append(i)
        best_idx = min(cands, key=lambda k: abs(self._timestamps[k] - t_sec))
        if abs(self._timestamps[best_idx] - t_sec) > max_dt_sec:
            return None
        return self._poses_by_ts[best_idx]

    def all_poses(self) -> List[Pose]:
        """Return all poses in timestamp order.  Used for canvas-bounds pre-scan."""
        return list(self._poses_by_ts)


# ══════════════════════════════════════════════════════════════════════════════
# Warp construction from pose
# ══════════════════════════════════════════════════════════════════════════════


def _build_warp_from_pose(
    pose: Pose,
    intrinsics: CameraIntrinsics,
    coord_cfg: CoordinateConfig,
    canvas_world_origin: Tuple[float, float],
    canvas_world_top: float,
    ppm: float,
    yaw_override: Optional[float] = None,
) -> np.ndarray:
    """Construct the 3×3 similarity homography mapping image pixels of a frame
    captured at `pose` to canvas pixels.

    Math (five stages composed right-to-left, since H = (last) @ ... @ (first))
    ──────────────────────────────────────────────────────────────────────────
      T_center   image → image with principal point at origin
      S_local    image pixels → drone-local metres at altitude z
                   (with image_x_sign, image_y_sign applied so axes align
                    with world directions when yaw=0)
      R_yaw      drone-local metres → world metres (rotates by effective yaw)
      T_drone    add drone world position (wx, wy)
      W_to_C     world metres → canvas pixels (apply ppm, flip y if requested)

    Yaw composition
    ───────────────
    effective_yaw = yaw_sign * (yaw_override or pose.yaw) + yaw_offset_rad

    Use `yaw_override` to supply a refined yaw without mutating `pose`.
    """
    yaw_csv = pose.yaw if yaw_override is None else yaw_override
    yaw_eff = coord_cfg.yaw_sign * yaw_csv + coord_cfg.yaw_offset_rad

    fx, fy = intrinsics.fx, intrinsics.fy
    cx_img, cy_img = intrinsics.cx, intrinsics.cy

    # 1. Centre on principal point.
    T_center = np.array(
        [[1.0, 0.0, -cx_img],
         [0.0, 1.0, -cy_img],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # 2. Image-pixel → drone-local metres (with axis-sign flips).
    sx = float(coord_cfg.image_x_sign) * pose.z / fx
    sy = float(coord_cfg.image_y_sign) * pose.z / fy
    S_local = np.array(
        [[sx,  0.0, 0.0],
         [0.0, sy,  0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # 3. Rotate by effective yaw (drone-local → world).
    c, s = math.cos(yaw_eff), math.sin(yaw_eff)
    R_yaw = np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # 4. Translate to drone world position.
    T_drone = np.array(
        [[1.0, 0.0, pose.x],
         [0.0, 1.0, pose.y],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # 5. World metres → canvas pixels (with optional y-flip).
    ox, oy_min = canvas_world_origin     # world (x, y) at canvas pixel (0, 0)*
    if coord_cfg.canvas_y_flip:
        # canvas_x = (X - ox) * ppm
        # canvas_y = (canvas_world_top - Y) * ppm     ← y flipped
        W_to_C = np.array(
            [[ppm,  0.0, -ox * ppm],
             [0.0, -ppm,  canvas_world_top * ppm],
             [0.0,  0.0,  1.0]],
            dtype=np.float64,
        )
    else:
        # canvas_x = (X - ox) * ppm
        # canvas_y = (Y - oy_min) * ppm     ← no flip
        W_to_C = np.array(
            [[ppm, 0.0, -ox * ppm],
             [0.0, ppm, -oy_min * ppm],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    H = W_to_C @ T_drone @ R_yaw @ S_local @ T_center
    return H


def _compute_canvas_bounds(
    poses: List[Pose],
    intrinsics: CameraIntrinsics,
    frame_w: int,
    frame_h: int,
    margin_factor: float = 1.1,
) -> Tuple[Tuple[float, float], float, Tuple[float, float, float, float]]:
    """Scan the CSV pose list to compute the canvas extent in world metres.

    The bounding box of all drone positions is enlarged by half the maximum
    frame footprint (at max altitude) plus a small safety factor, so any
    frame's full image area fits within the canvas regardless of which way
    the drone was facing.

    Returns
    -------
        (origin_world_xy, world_top_y, (wx_min, wx_max, wy_min, wy_max))

    origin_world_xy : (ox, oy_min) — world (x, y) at canvas pixel (0, 0)
                      when canvas_y_flip = False.
    world_top_y     : the max world y, used as the top-of-canvas reference
                      when canvas_y_flip = True.
    bbox            : raw drone-position bounding box, for diagnostics.
    """
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    zs = [p.z for p in poses]
    wx_min, wx_max = min(xs), max(xs)
    wy_min, wy_max = min(ys), max(ys)
    z_max = max(zs)

    # Half-diagonal of the largest frame footprint, in metres.  Worst case
    # is the longer image axis at max altitude.
    half_x = (frame_w / 2.0) * z_max / intrinsics.fx
    half_y = (frame_h / 2.0) * z_max / intrinsics.fy
    half_diag = math.hypot(half_x, half_y)

    margin = half_diag * margin_factor

    ox = wx_min - margin
    ox_max = wx_max + margin
    oy_min = wy_min - margin
    oy_max = wy_max + margin

    return (ox, oy_min), oy_max, (wx_min, wx_max, wy_min, wy_max)


# ══════════════════════════════════════════════════════════════════════════════
# Yaw refinement (bounded 1D rotation correction)
# ══════════════════════════════════════════════════════════════════════════════


def _refine_yaw_against_canvas(
    canvas: np.ndarray,
    img_kp,
    img_des,
    H_initial: np.ndarray,
    drone_canvas_pos: Tuple[float, float],
    frame_w: int,
    frame_h: int,
    detector,
    norm: int,
    feature_mask_builder=None,
    img_for_mask: Optional[np.ndarray] = None,
    max_correction_rad: float = math.radians(5.0),
    min_matches: int = 12,
    match_ratio: float = 0.70,
) -> Tuple[Optional[float], int]:
    """Solve a closed-form bounded rotation that aligns this frame to the
    canvas content already under its footprint.

    Returns (delta_canvas_rad, n_inliers).  delta_canvas_rad is the rotation
    in *canvas* angle space — the caller must convert to a yaw correction by
    multiplying by yaw_sign.

    Returns (None, 0) if any of:
      • the canvas ROI under the frame is empty (nothing to match against)
      • too few features in either set
      • too few mutual-cross-check matches
      • MAD outlier-rejection leaves too few inliers
      • the recovered |Δθ| exceeds max_correction_rad

    Algorithm
    ─────────
      1. Warp the four frame corners by H_initial → compute the canvas ROI.
      2. Extract that ROI; if it has any non-zero content, detect features.
      3. Match canvas-ROI features against img features (mutual cross-check).
      4. For each matched pair, warp the image keypoint by H_initial to put
         it in canvas pixel space.  Subtract the drone canvas position from
         both points so the rotation is around the drone, not the origin.
      5. MAD-filter outliers on the per-pair angle delta.
      6. Closed-form Δθ = atan2(Σ p'×q', Σ p'·q')  (the 2D Wahba solution).

    Translation and scale are held constant (the refinement is geometrically
    a pure rotation around the drone's canvas position); the OptiTrack
    position is never overridden.
    """
    if img_des is None or len(img_kp) < min_matches:
        return None, 0

    canvas_h, canvas_w = canvas.shape[:2]

    # ── 1. Canvas ROI = bounding box of warped frame corners ─────────────────
    corners = np.float32(
        [[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]]
    ).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H_initial).reshape(-1, 2)

    x0 = max(0, int(math.floor(wc[:, 0].min())))
    y0 = max(0, int(math.floor(wc[:, 1].min())))
    x1 = min(canvas_w, int(math.ceil(wc[:, 0].max())) + 1)
    y1 = min(canvas_h, int(math.ceil(wc[:, 1].max())) + 1)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None, 0   # frame barely intersects canvas; nothing to match

    canvas_roi = canvas[y0:y1, x0:x1]
    if not bool((canvas_roi.sum(axis=2) > 0).any()):
        return None, 0   # ROI is entirely blank — no prior content to align to

    # ── 2. Detect features on the canvas ROI ────────────────────────────────
    # Use the same exclusion-mask convention if the caller provided a builder.
    # Note: the canvas may contain colour-masked content (e.g. yellow → white),
    # which actually helps feature matching by removing distracting colour
    # information; the underlying texture is what aligns.
    roi_mask = None
    if feature_mask_builder is not None:
        roi_mask = feature_mask_builder(canvas_roi)
    kp_c, des_c = _kp_des(detector, canvas_roi, mask=roi_mask)
    if des_c is None or len(kp_c) < min_matches:
        return None, 0

    # ── 3. Mutual-cross-check matching ──────────────────────────────────────
    matches = _match(img_des, des_c, norm, ratio=match_ratio, mutual=True)
    if len(matches) < min_matches:
        return None, 0

    # ── 4. Put both sides into canvas pixel space, centred on drone pos ─────
    pts_img    = np.float32([img_kp[m.queryIdx].pt for m in matches])
    pts_canvas_local = np.float32([kp_c[m.trainIdx].pt for m in matches])
    pts_canvas = pts_canvas_local + np.array([x0, y0], dtype=np.float32)

    pts_img_in_canvas = cv2.perspectiveTransform(
        pts_img.reshape(-1, 1, 2), H_initial
    ).reshape(-1, 2)

    dx, dy = drone_canvas_pos
    p = pts_img_in_canvas - np.array([dx, dy], dtype=np.float32)   # current
    q = pts_canvas        - np.array([dx, dy], dtype=np.float32)   # target

    # Drop points too close to the drone position (their angle is ill-defined).
    r_p = np.linalg.norm(p, axis=1)
    r_q = np.linalg.norm(q, axis=1)
    keep = (r_p > 5.0) & (r_q > 5.0)
    if int(keep.sum()) < min_matches:
        return None, 0
    p, q = p[keep], q[keep]

    # ── 5. MAD outlier rejection on per-pair angle delta ────────────────────
    ang_p = np.arctan2(p[:, 1], p[:, 0])
    ang_q = np.arctan2(q[:, 1], q[:, 0])
    delta = np.arctan2(np.sin(ang_q - ang_p), np.cos(ang_q - ang_p))   # in (-π, π]
    med = float(np.median(delta))
    mad = float(np.median(np.abs(delta - med))) + 1e-4
    keep2 = np.abs(delta - med) < 5.0 * mad
    if int(keep2.sum()) < min_matches:
        return None, 0
    p, q = p[keep2], q[keep2]

    # ── 6. Closed-form Wahba 2D rotation ────────────────────────────────────
    s_sum = float(np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]))
    c_sum = float(np.sum(p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]))
    delta_canvas = math.atan2(s_sum, c_sum)

    if abs(delta_canvas) > max_correction_rad:
        return None, int(keep2.sum())

    return delta_canvas, int(keep2.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CSVStitchConfig:
    """All knobs for CSV-driven reconstruction."""

    # ── REQUIRED ──────────────────────────────────────────────────────────────
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics.from_fov(1920, 1080, h_fov_deg=70.0)
    )
    """Camera intrinsics.  YOU MUST CONFIGURE FOR YOUR CAMERA.
    The default is a guess for a 1920×1080 frame with 70° horizontal FOV."""

    coord_cfg: CoordinateConfig = field(default_factory=CoordinateConfig)
    """Sign / orientation conventions.  See CoordinateConfig docstring."""

    # ── canvas sizing ─────────────────────────────────────────────────────────
    ppm: Optional[float] = None
    """Canvas resolution in pixels per metre.  None = auto-pick so the canvas
    matches source resolution at average altitude (one canvas pixel ≈ one
    image pixel of overhead detail).  Override if you want a specific scale
    or are hitting max_canvas_px."""

    margin_factor: float = 1.1
    """Multiplier on the half-frame-footprint margin added around the
    drone-position bounding box when sizing the canvas.  1.0 = exactly fits
    every frame; 1.1 leaves 10% slack so a frame near the boundary doesn't
    clip."""

    max_canvas_px: int = 8000
    """Hard cap on the largest canvas dimension (pixels).  If the auto-ppm
    would produce a canvas larger than this, ppm is reduced.  Default 8000
    px = up to ~183 MB canvas."""

    # ── frame quality gates ───────────────────────────────────────────────────
    blur_thresh: float = 50.0
    """Laplacian variance below this = blurry, drop the frame."""

    artifact_thresh: float = 1.5
    """DCT block-artifact ratio above this = codec-damaged, drop the frame."""

    lo_brightness: float = 15.0
    hi_brightness: float = 240.0
    """Mean grayscale must fall in (lo, hi)."""

    # ── static-frame skipping (uses CSV deltas, no feature matching) ─────────
    min_movement_m: float = 0.02
    """If the drone has moved less than this many metres in (x, y) since the
    last placed frame, skip — the new frame would just paint over identical
    content.  Set to 0 to disable static skipping."""

    min_yaw_change_rad: float = math.radians(2.0)
    """Or, if the drone moved less than min_movement_m but its yaw changed by
    more than this much, KEEP the frame (rotation alone changes coverage)."""

    # ── sampling ─────────────────────────────────────────────────────────────
    target_fps: float = 8.0
    """Frames-per-second sampling rate from the source video."""

    processing_scale: float = 1.0
    """Optional pre-warp downsample of each frame.  0.5 = quarter the data."""

    # ── lens undistortion (optional) ─────────────────────────────────────────
    undistort_frames: bool = False
    """If True AND intrinsics.dist_coeffs is provided, apply lens undistortion
    to every frame before colour masking and warping.

    Default is False — frames are used as-is.  Set True when you have a
    significantly distorted lens (fisheye, wide-angle) AND have measured
    the distortion coefficients via cv2.calibrateCamera; the radial barrel
    that bends the canvas at frame edges goes away.

    Implementation: cv2.initUndistortRectifyMap is called once at startup
    to build the (x, y) → (x', y') lookup tables for the processing-frame
    size, then cv2.remap applies them per frame.  No allocation in the
    hot loop; the maps add ~16 MB for a 1920×1080 frame.

    If undistort_frames=True but dist_coeffs is None, a warning is printed
    and frames are passed through unchanged (graceful fallback)."""

    # ── colour masking (visual, pre-blend) ───────────────────────────────────
    color_masks: List[ColorRangeMask] = field(default_factory=list)
    """HSV colour ranges replaced with a fixed colour BEFORE blending onto
    the canvas.  Does not affect feature detection.  Same semantics as
    drone_map_gen.ReconstructConfig.color_masks."""

    # ── feature detection (yaw refinement only) ──────────────────────────────
    feature_exclude_hsv: List[ColorRangeMask] = field(default_factory=list)
    """HSV ranges where features are NOT detected (dilation + invert).  Use
    to suppress the periodic blue grid when running yaw refinement; the
    placement itself never uses features."""

    feature_exclude_dilate_px: int = 5
    """Dilation applied to the feature-exclusion mask (pixels)."""

    # ── yaw refinement ───────────────────────────────────────────────────────
    refine_yaw: bool = False
    """If True, run per-frame yaw refinement against the existing canvas
    content.  Falls back silently to CSV yaw if matching fails."""

    refine_yaw_max_correction_deg: float = 5.0
    """Maximum |Δyaw| applied per frame (degrees).  Larger corrections are
    treated as matching errors and rejected."""

    refine_yaw_min_matches: int = 12
    """Minimum mutual-cross-check matches required to attempt a refinement."""

    refine_yaw_match_ratio: float = 0.70
    """Lowe's ratio for the refinement matcher (also mutual cross-checked)."""

    # ── global yaw-offset calibration ────────────────────────────────────────
    calibrate_yaw_offset: bool = False
    """If True, run a calibration pass over the first calibration_n_frames
    successfully-placed frames, collect the per-frame Δyaw corrections, and
    apply the median as a CONSTANT additive offset to yaw_offset_rad for
    the remainder of the run.  Captures a fixed IMU-to-world misalignment."""

    calibration_n_frames: int = 30
    """How many frames to use in the calibration pass before locking in the
    offset.  Should be enough to span a few different yaw values but small
    compared to the full flight."""

    # ── blending ─────────────────────────────────────────────────────────────
    blend_mode: str = "pyramid"
    """"feather" — distance-weighted linear blend (fast)
    "pyramid"  — Laplacian multi-band blend (best quality, slower)
    "flat"     — 50/50 average (debug)"""

    pyramid_levels: int = 4
    """Laplacian pyramid depth (only used when blend_mode='pyramid')."""

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_frame_id_offset: int = 1
    """CSV frame_id starts at 1 by convention; OpenCV frame index starts at 0.
    Default offset of 1 makes CSV frame_id = (video_frame_idx + 1)."""

    csv_use_timestamps: bool = False
    """If True, look up CSV rows by timestamp (nearest within max_dt_sec)
    instead of frame_id.  Use when the CSV is at a different rate than the
    video."""

    csv_max_dt_sec: float = 0.05
    """Maximum timestamp gap accepted for nearest-row lookup."""


# ══════════════════════════════════════════════════════════════════════════════
# Reconstructor
# ══════════════════════════════════════════════════════════════════════════════


class CSVMapReconstructor:
    """Stitches a top-down drone map using CSV poses for placement and
    (optionally) feature-based yaw refinement on top.

    Memory model
    ────────────
        Canvas       ~ max_canvas_px² × 3 bytes  (e.g. 8000² × 3 ≈ 183 MB)
        Per-frame    ~ 30–40 MB transient (decoded BGR + colour-masked copy
                      + grayscale + warped ROI + optional canvas-ROI for
                      yaw refinement)
        No descriptor caches, no keyframe ring buffer.  No allocation spikes.
    """

    def __init__(self, cfg: CSVStitchConfig):
        self.cfg = cfg
        self.detector, self.norm = _make_detector()

        self._canvas: Optional[np.ndarray] = None
        self._canvas_origin_world: Tuple[float, float] = (0.0, 0.0)
        self._canvas_world_top: float = 0.0
        self._ppm: float = 0.0

        # Live yaw offset (mutable — modified by calibration pass).
        self._yaw_offset_rad: float = cfg.coord_cfg.yaw_offset_rad

        # Tracking for static-frame skip and stats.
        self._last_placed_pose: Optional[Pose] = None
        self._n_placed = 0
        self._n_skipped_csv = 0
        self._n_skipped_quality = 0
        self._n_skipped_static = 0
        self._n_yaw_refined = 0
        self._n_yaw_fallback = 0

        # Undistortion remap tables, built once in add_video() when
        # cfg.undistort_frames is enabled and dist_coeffs is provided.
        # None means "no undistortion" — frames pass through unchanged.
        self._undistort_map1: Optional[np.ndarray] = None
        self._undistort_map2: Optional[np.ndarray] = None

    # ── public API ───────────────────────────────────────────────────────────

    def add_video(self, video_path: str, csv_path: str, verbose: bool = True) -> None:
        """End-to-end pipeline: load CSV, size canvas, stream + place all
        sampled frames.  Handles the optional yaw-offset calibration pass."""
        poses = PoseTable(csv_path, frame_id_offset=self.cfg.csv_frame_id_offset)
        all_poses = poses.all_poses()
        if not all_poses:
            raise ValueError(f"CSV {csv_path!r} contains no rows")

        # Probe the first video frame to learn the frame size.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path!r}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, probe = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"Cannot read first frame of {video_path!r}")
        full_h, full_w = probe.shape[:2]
        cap.release()

        proc_w = max(1, int(full_w * self.cfg.processing_scale))
        proc_h = max(1, int(full_h * self.cfg.processing_scale))
        # Intrinsics rescale with processing_scale: a downsampled image has a
        # proportionally smaller focal length in pixels (same FOV, fewer pixels).
        intr = self._scaled_intrinsics(self.cfg.processing_scale)

        if verbose:
            print(f"  source: {src_fps:.1f} fps, {n_total} frames, {full_w}×{full_h}")
            print(f"  processing: {proc_w}×{proc_h} at scale {self.cfg.processing_scale}")
            print(f"  CSV: {len(all_poses)} pose rows, "
                  f"frame_id range {all_poses[0].frame_id}–{all_poses[-1].frame_id}")

        # ── build undistortion remap tables (once, if requested) ─────────────
        # When cfg.undistort_frames=True AND intr.dist_coeffs is provided, we
        # precompute the (x, y) → (x', y') lookup tables via
        # initUndistortRectifyMap and apply them with cv2.remap inside the
        # per-frame loop.  Using the SAME camera matrix as both the source
        # and destination means the resulting undistorted image keeps the
        # original fx/fy/cx/cy values — so the rest of the warp math is
        # unchanged.  (Distortion coefficients are dimensionless and remain
        # valid at any processing_scale.)
        if self.cfg.undistort_frames:
            if intr.dist_coeffs is None:
                if verbose:
                    print("  [warn] undistort_frames=True but "
                          "intrinsics.dist_coeffs is None — passing frames through")
            else:
                K = np.array(
                    [[intr.fx, 0.0,     intr.cx],
                     [0.0,     intr.fy, intr.cy],
                     [0.0,     0.0,     1.0]],
                    dtype=np.float64,
                )
                D = np.array(intr.dist_coeffs, dtype=np.float64)
                self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
                    K, D, R=None, newCameraMatrix=K,
                    size=(proc_w, proc_h), m1type=cv2.CV_16SC2,
                )
                if verbose:
                    coeffs_str = ", ".join(f"{c:+.4f}" for c in D.flat)
                    print(f"  undistortion ON  |  dist_coeffs=[{coeffs_str}]")

        # Compute canvas extent from world bounds.
        origin, top, bbox = _compute_canvas_bounds(
            all_poses, intr, proc_w, proc_h, margin_factor=self.cfg.margin_factor
        )
        self._canvas_origin_world = origin
        self._canvas_world_top = top
        world_w = (bbox[1] - bbox[0]) + 2 * (top - bbox[3])  # approximate
        # ppm: auto = match image resolution at average altitude
        if self.cfg.ppm is None:
            avg_z = float(np.mean([p.z for p in all_poses]))
            ppm_auto = intr.fx / max(avg_z, 1e-3)
            self._ppm = ppm_auto
        else:
            self._ppm = float(self.cfg.ppm)

        # Cap canvas size if necessary.
        canvas_w_m = (bbox[1] - bbox[0]) + 2 * (top - bbox[3])  # raw width margin
        # Recompute exact world extents covered by the chosen origin:
        ox, oy_min = origin
        world_extent_x = (bbox[1] + (top - bbox[3])) - ox     # right - left
        world_extent_y = (top) - oy_min                        # top - bottom
        max_extent = max(world_extent_x, world_extent_y)
        canvas_px_needed = int(math.ceil(max_extent * self._ppm))
        if canvas_px_needed > self.cfg.max_canvas_px:
            new_ppm = self.cfg.max_canvas_px / max_extent
            if verbose:
                print(f"  ppm reduced {self._ppm:.1f} → {new_ppm:.1f} "
                      f"to fit max_canvas_px={self.cfg.max_canvas_px}")
            self._ppm = new_ppm
            canvas_px_needed = self.cfg.max_canvas_px

        canvas_w = int(math.ceil(world_extent_x * self._ppm))
        canvas_h = int(math.ceil(world_extent_y * self._ppm))
        self._canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        if verbose:
            mb = canvas_h * canvas_w * 3 / 1_048_576
            print(f"  canvas: {canvas_w}×{canvas_h} px  ({mb:.0f} MB)  "
                  f"ppm={self._ppm:.1f}")
            print(f"  drone bbox: x [{bbox[0]:.2f}, {bbox[1]:.2f}]  "
                  f"y [{bbox[2]:.2f}, {bbox[3]:.2f}]  "
                  f"z_max={max(p.z for p in all_poses):.2f}")

        # ── pass 1 (optional): yaw-offset calibration ────────────────────────
        if self.cfg.calibrate_yaw_offset:
            if verbose:
                print(f"\n  ── yaw-offset calibration pass "
                      f"(first {self.cfg.calibration_n_frames} placed frames) ──")
            self._calibration_pass(
                video_path=video_path,
                poses=poses,
                src_fps=src_fps,
                full_w=full_w,
                full_h=full_h,
                intr=intr,
                verbose=verbose,
            )
            # Discard the partial canvas built during calibration; main pass
            # rebuilds it cleanly with the corrected yaw offset.
            self._canvas[:] = 0
            self._n_placed = 0
            self._last_placed_pose = None
            if verbose:
                print(f"  yaw_offset_rad locked at: "
                      f"{self._yaw_offset_rad:+.4f} rad "
                      f"({math.degrees(self._yaw_offset_rad):+.2f}°)\n")

        # ── pass 2: main reconstruction ──────────────────────────────────────
        if verbose:
            print(f"  ── main reconstruction pass ──")
        self._main_pass(
            video_path=video_path,
            poses=poses,
            src_fps=src_fps,
            full_w=full_w,
            full_h=full_h,
            intr=intr,
            verbose=verbose,
        )

    def get_map(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        crop: bool = True,
    ) -> np.ndarray:
        """Return the stitched map.  See drone_map_gen.MapReconstructor.get_map
        for crop / output_shape semantics — identical here."""
        if self._canvas is None:
            raise RuntimeError("No frames placed yet.")
        canvas = self._canvas
        self._canvas = None

        if crop:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            nz   = cv2.findNonZero(gray)
            del gray
            if nz is not None:
                x, y, cw, ch = cv2.boundingRect(nz)
                canvas = canvas[y:y + ch, x:x + cw].copy()

        if output_shape is not None:
            result = cv2.resize(canvas, output_shape, interpolation=cv2.INTER_LANCZOS4)
            del canvas
            return result
        return canvas

    @property
    def stats(self) -> dict:
        ch, cw = self._canvas.shape[:2] if self._canvas is not None else (0, 0)
        return {
            "placed":               self._n_placed,
            "skipped_csv":          self._n_skipped_csv,
            "skipped_quality":      self._n_skipped_quality,
            "skipped_static":       self._n_skipped_static,
            "yaw_refined":          self._n_yaw_refined,
            "yaw_fallback":         self._n_yaw_fallback,
            "canvas_hw":            (ch, cw),
            "canvas_mb":            round(ch * cw * 3 / 1_048_576, 1),
            "yaw_offset_rad":       round(self._yaw_offset_rad, 6),
            "yaw_offset_deg":       round(math.degrees(self._yaw_offset_rad), 3),
        }

    # ── internal: scaled intrinsics ──────────────────────────────────────────

    def _scaled_intrinsics(self, scale: float) -> CameraIntrinsics:
        if scale == 1.0:
            return self.cfg.intrinsics
        i = self.cfg.intrinsics
        # Distortion coefficients are dimensionless (operate on normalised
        # coordinates), so they remain valid when fx/fy/cx/cy are scaled.
        return CameraIntrinsics(
            fx=i.fx * scale, fy=i.fy * scale,
            cx=i.cx * scale, cy=i.cy * scale,
            dist_coeffs=i.dist_coeffs,
        )

    # ── internal: passes ─────────────────────────────────────────────────────

    def _calibration_pass(
        self,
        video_path: str,
        poses: PoseTable,
        src_fps: float,
        full_w: int,
        full_h: int,
        intr: CameraIntrinsics,
        verbose: bool,
    ) -> None:
        """Place the first N frames with refinement forced ON, collect per-
        frame Δyaw, take the median, and lock that in as yaw_offset_rad."""
        deltas: List[float] = []
        target_count = self.cfg.calibration_n_frames

        for placed_pose, delta_canvas in self._stream_and_place(
            video_path, poses, src_fps, full_w, full_h, intr,
            force_refine=True, stop_after_placed=target_count, verbose=verbose,
        ):
            if delta_canvas is not None:
                deltas.append(delta_canvas)

        if not deltas:
            if verbose:
                print("  [calibration] no refinements succeeded; "
                      "yaw_offset_rad unchanged")
            return

        # delta_canvas is in *canvas* rotation space.  The CSV yaw enters the
        # warp as (yaw_sign * yaw + yaw_offset_rad), so the offset that
        # corrects all future frames is exactly the median delta_canvas.
        median_delta = float(np.median(deltas))
        if verbose:
            print(f"  [calibration] {len(deltas)} successful refinements")
            print(f"  [calibration] median Δ = {median_delta:+.4f} rad "
                  f"({math.degrees(median_delta):+.2f}°)")
        self._yaw_offset_rad = self.cfg.coord_cfg.yaw_offset_rad + median_delta

    def _main_pass(
        self,
        video_path: str,
        poses: PoseTable,
        src_fps: float,
        full_w: int,
        full_h: int,
        intr: CameraIntrinsics,
        verbose: bool,
    ) -> None:
        for _ in self._stream_and_place(
            video_path, poses, src_fps, full_w, full_h, intr,
            force_refine=False, stop_after_placed=None, verbose=verbose,
        ):
            pass

    # ── internal: the actual frame loop (shared by both passes) ──────────────

    def _stream_and_place(
        self,
        video_path: str,
        poses: PoseTable,
        src_fps: float,
        full_w: int,
        full_h: int,
        intr: CameraIntrinsics,
        force_refine: bool,
        stop_after_placed: Optional[int],
        verbose: bool,
    ):
        """Generator that yields (pose, delta_canvas_rad_or_None) per placed
        frame.  Centralises the per-frame logic so calibration and main pass
        share exactly the same placement code."""
        do_refine = force_refine or self.cfg.refine_yaw
        step = max(1, int(round(src_fps / self.cfg.target_fps)))
        max_corr_rad = math.radians(self.cfg.refine_yaw_max_correction_deg)

        cap = cv2.VideoCapture(video_path)
        try:
            frame_idx = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % step != 0:
                    continue

                # ── CSV lookup ──────────────────────────────────────────────
                if self.cfg.csv_use_timestamps:
                    t_sec = frame_idx / src_fps
                    pose = poses.by_timestamp(t_sec, self.cfg.csv_max_dt_sec)
                else:
                    pose = poses.by_frame_id(frame_idx)
                if pose is None:
                    self._n_skipped_csv += 1
                    if verbose:
                        print(f"  [skip:csv] frame {frame_idx}: no matching pose row")
                    continue

                # ── optional downscale ──────────────────────────────────────
                if self.cfg.processing_scale != 1.0:
                    fw = max(1, int(frame.shape[1] * self.cfg.processing_scale))
                    fh = max(1, int(frame.shape[0] * self.cfg.processing_scale))
                    frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
                fh, fw = frame.shape[:2]

                # ── quality gate ────────────────────────────────────────────
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                qok, qreason = _assess_frame(
                    frame,
                    blur_thresh=self.cfg.blur_thresh,
                    artifact_thresh=self.cfg.artifact_thresh,
                    gray=gray,
                    lo_brightness=self.cfg.lo_brightness,
                    hi_brightness=self.cfg.hi_brightness,
                )
                del gray
                if not qok:
                    self._n_skipped_quality += 1
                    if verbose:
                        print(f"  [skip:quality] frame {frame_idx}: {qreason}")
                    continue

                # ── static-frame skip (CSV-based) ───────────────────────────
                if self._last_placed_pose is not None and self.cfg.min_movement_m > 0:
                    dx = pose.x - self._last_placed_pose.x
                    dy = pose.y - self._last_placed_pose.y
                    d_xy = math.hypot(dx, dy)
                    dyaw = abs(_wrap_pi(pose.yaw - self._last_placed_pose.yaw))
                    if (d_xy < self.cfg.min_movement_m
                            and dyaw < self.cfg.min_yaw_change_rad):
                        self._n_skipped_static += 1
                        if verbose:
                            print(f"  [skip:static] frame {frame_idx}: "
                                  f"dxy={d_xy:.3f} m  dyaw={math.degrees(dyaw):.2f}°")
                        continue

                # ── lens undistortion (if remap tables were built) ──────────
                # Applied AFTER quality and static gates so dropped frames
                # don't pay for the remap; applied BEFORE colour masking and
                # feature detection so both operate on rectified geometry.
                if self._undistort_map1 is not None:
                    frame = cv2.remap(
                        frame,
                        self._undistort_map1, self._undistort_map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                # ── colour mask the frame for blending ──────────────────────
                # Note: we keep the unmasked frame for feature detection during
                # yaw refinement, since masked white pixels hurt SIFT.
                img_stitch = _apply_color_masks(frame, self.cfg.color_masks)

                # ── build initial warp from CSV pose ────────────────────────
                H = _build_warp_from_pose(
                    pose=pose,
                    intrinsics=intr,
                    coord_cfg=self._effective_coord_cfg(),
                    canvas_world_origin=self._canvas_origin_world,
                    canvas_world_top=self._canvas_world_top,
                    ppm=self._ppm,
                )

                # ── optional: yaw refinement ────────────────────────────────
                delta_canvas: Optional[float] = None
                if do_refine and self._n_placed > 0:
                    # Build the feature-exclusion mask FROM the unmasked frame.
                    feat_mask = _make_feature_mask(
                        frame,
                        self.cfg.feature_exclude_hsv,
                        dilate_px=self.cfg.feature_exclude_dilate_px,
                    )
                    img_kp, img_des = _kp_des(self.detector, frame, mask=feat_mask)
                    if feat_mask is not None:
                        del feat_mask

                    # Drone position in canvas pixels (translation row of H @ centre).
                    drone_canvas = cv2.perspectiveTransform(
                        np.float32([[[intr.cx, intr.cy]]]), H
                    )[0][0]

                    # Builder lambda passes the same HSV exclusion to canvas-ROI features.
                    def _builder(roi_img):
                        return _make_feature_mask(
                            roi_img,
                            self.cfg.feature_exclude_hsv,
                            dilate_px=self.cfg.feature_exclude_dilate_px,
                        )

                    delta_canvas, n_in = _refine_yaw_against_canvas(
                        canvas=self._canvas,
                        img_kp=img_kp,
                        img_des=img_des,
                        H_initial=H,
                        drone_canvas_pos=(float(drone_canvas[0]), float(drone_canvas[1])),
                        frame_w=fw,
                        frame_h=fh,
                        detector=self.detector,
                        norm=self.norm,
                        feature_mask_builder=_builder,
                        img_for_mask=frame,
                        max_correction_rad=max_corr_rad,
                        min_matches=self.cfg.refine_yaw_min_matches,
                        match_ratio=self.cfg.refine_yaw_match_ratio,
                    )

                    if delta_canvas is not None:
                        # Convert canvas rotation Δ to CSV-yaw Δ via yaw_sign,
                        # then rebuild H with the refined yaw.
                        yaw_refined = pose.yaw + (
                            self.cfg.coord_cfg.yaw_sign * delta_canvas
                        )
                        H = _build_warp_from_pose(
                            pose=pose,
                            intrinsics=intr,
                            coord_cfg=self._effective_coord_cfg(),
                            canvas_world_origin=self._canvas_origin_world,
                            canvas_world_top=self._canvas_world_top,
                            ppm=self._ppm,
                            yaw_override=yaw_refined,
                        )
                        self._n_yaw_refined += 1
                        if verbose:
                            print(f"  [yaw-refine] frame {frame_idx}: "
                                  f"Δ={math.degrees(delta_canvas):+.2f}° "
                                  f"(n_in={n_in})")
                    else:
                        self._n_yaw_fallback += 1

                # ── warp + blend into canvas ────────────────────────────────
                self._warp_and_blend_roi(img_stitch, H)
                self._n_placed += 1
                self._last_placed_pose = pose

                yield pose, delta_canvas

                if stop_after_placed is not None and self._n_placed >= stop_after_placed:
                    return

                if verbose and self._n_placed % 50 == 0:
                    print(f"  ... {self._n_placed} frames placed")

        finally:
            cap.release()

    def _effective_coord_cfg(self) -> CoordinateConfig:
        """Return a CoordinateConfig with the LIVE yaw_offset_rad value
        (which may differ from cfg.coord_cfg if calibration ran)."""
        c = self.cfg.coord_cfg
        return CoordinateConfig(
            image_x_sign=c.image_x_sign,
            image_y_sign=c.image_y_sign,
            yaw_sign=c.yaw_sign,
            yaw_offset_rad=self._yaw_offset_rad,
            canvas_y_flip=c.canvas_y_flip,
        )

    # ── internal: warp + blend (same memory-efficient ROI approach) ──────────

    def _warp_and_blend_roi(self, img: np.ndarray, H: np.ndarray) -> None:
        """ROI-based warp + blend (frame-sized scratch, not canvas-sized).
        Same algorithm as drone_map_gen.MapReconstructor._warp_and_blend_roi
        — the only function that touches canvas memory besides initialisation."""
        fh, fw = img.shape[:2]
        ch, cw = self._canvas.shape[:2]

        corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        x0 = max(0, int(math.floor(wc[:, 0].min())))
        y0 = max(0, int(math.floor(wc[:, 1].min())))
        x1 = min(cw, int(math.ceil(wc[:, 0].max())) + 1)
        y1 = min(ch, int(math.ceil(wc[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        roi_w, roi_h = x1 - x0, y1 - y0
        T_shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        H_roi = T_shift @ H

        warped_roi = cv2.warpPerspective(img, H_roi, (roi_w, roi_h))
        mask_new = warped_roi.sum(axis=2) > 0

        canvas_roi = self._canvas[y0:y1, x0:x1]
        mode = self.cfg.blend_mode

        if mode == "flat":
            mask_c = canvas_roi.sum(axis=2) > 0
            only_new = mask_new & ~mask_c
            overlap  = mask_new &  mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                canvas_roi[overlap] = (
                    canvas_roi[overlap].astype(np.float32) * 0.5
                    + warped_roi[overlap].astype(np.float32) * 0.5
                ).astype(np.uint8)

        elif mode == "pyramid":
            mask_c = canvas_roi.sum(axis=2) > 0
            only_new = mask_new & ~mask_c
            overlap  = mask_new &  mask_c
            canvas_roi[only_new] = warped_roi[only_new]
            if overlap.any():
                dist = cv2.distanceTransform(
                    overlap.astype(np.uint8) * 255, cv2.DIST_L2, 5
                )
                d_max = dist.max()
                alpha = (dist / d_max if d_max > 0
                         else np.full_like(dist, 0.5)).astype(np.float32)
                blended = _laplacian_pyramid_blend(
                    canvas_roi, warped_roi, alpha, self.cfg.pyramid_levels
                )
                canvas_roi[overlap] = blended[overlap]

        else:   # "feather" (default elsewhere; pyramid is default here)
            blended = _feather_blend_roi(canvas_roi, warped_roi, mask_new)
            self._canvas[y0:y1, x0:x1] = blended


# ══════════════════════════════════════════════════════════════════════════════
# Small utility
# ══════════════════════════════════════════════════════════════════════════════


def _wrap_pi(a: float) -> float:
    """Wrap an angle (radians) into (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# ══════════════════════════════════════════════════════════════════════════════
# Top-level convenience
# ══════════════════════════════════════════════════════════════════════════════


def reconstruct_from_csv(
    video_path: str,
    csv_path: str,
    cfg: CSVStitchConfig,
    output_shape: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """One-call API: load CSV + video, build map, optionally save, return it."""
    sep = "─" * 60
    if verbose:
        print(sep)
        print(f"  CSV-driven reconstruction")
        print(f"    video : {video_path!r}")
        print(f"    csv   : {csv_path!r}")
        print(sep)

    rec = CSVMapReconstructor(cfg)
    rec.add_video(video_path, csv_path, verbose=verbose)

    if verbose:
        print()
        print(sep)
        print(f"  Finalising  |  {rec.stats}")
        print(sep)

    result = rec.get_map(output_shape=output_shape)

    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(out_dir, exist_ok=True)
        if not cv2.imwrite(save_path, result):
            raise IOError(f"cv2.imwrite failed for {save_path!r}")
        if verbose:
            print(f"  Saved → {save_path}")

    return result