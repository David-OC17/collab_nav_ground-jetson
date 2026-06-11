#!/usr/bin/env python3
"""
Compare two ArUco localisation approaches on the same video + telemetry.

  Approach A — Stitching (described)
    2D pixel geometry: deproject marker pixel offset to metres using drone
    altitude and focal length, rotate by yaw only, add to drone position.
    No solvePnP, no camera mounting offset, no roll/pitch compensation.
    Aggregation: per-axis median over all accepted observations.

  Approach B — Full pipeline (arena_marker_localizer)
    solvePnP(IPPE_SQUARE) with calibrated intrinsics, full 4-stage SE(3) chain
    (T_map_from_opti @ T_opti_from_drone @ T_drone_from_cam @ T_cam_from_marker),
    full roll+pitch+yaw from quaternion, reprojection-error gate, velocity gate,
    quality filter, MAD outlier gate, Weiszfeld geometric median.

Usage
─────
    python aruco_stitch_compare.py                        # defaults to scan29
    python aruco_stitch_compare.py --video V --csv C
    python aruco_stitch_compare.py --stride 2             # process every 2nd frame
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

# ── source-tree import (works without a colcon install) ───────────────────────
_PKG = os.path.join(os.path.dirname(__file__), "..", "src", "arena_marker_localizer")
sys.path.insert(0, os.path.abspath(_PKG))

from arena_marker_localizer.aggregation import AggregationConfig
from arena_marker_localizer.intrinsics import load_intrinsics
from arena_marker_localizer.marker_detection import DictionaryConfig
from arena_marker_localizer.optitrack import load_optitrack_csv
from arena_marker_localizer.pipeline import PipelineConfig, MarkerResult, run_pipeline
from arena_marker_localizer.quality import QualityConfig
from arena_marker_localizer.transforms import (
    OptiTrackAxisConfig, R_to_euler_zyx, StaticTransform6DoF,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCAN  = os.path.join(_ROOT, "src", "arena_map_builder", "data", "drone_scans", "scan29")
_CALIB = os.path.join(_ROOT, "src", "arena_marker_localizer", "config", "calibration.yaml")
_CFG   = os.path.join(_ROOT, "src", "arena_marker_localizer", "config", "default.yaml")
_GT    = os.path.join(_ROOT, "src", "arena_marker_localizer", "config", "aruco_pose_gt", "scan29.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_K(calib_path: str) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from an OpenCV-format calibration YAML."""
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    K  = fs.getNode("camera_matrix").mat()
    fs.release()
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _tmo_from_yaml(params: dict) -> np.ndarray:
    """Build T_map_from_opti (4×4) from the default.yaml params block."""
    tmo = params.get("T_map_from_opti", {})
    return StaticTransform6DoF(
        x=tmo.get("x", 0.0),    y=tmo.get("y", 0.0),    z=tmo.get("z", 0.0),
        roll=tmo.get("roll", 0.0), pitch=tmo.get("pitch", 0.0), yaw=tmo.get("yaw", 0.0),
    ).as_matrix()


def _cfg_from_yaml(calib_path: str, params: dict, stride: int) -> PipelineConfig:
    """Build a PipelineConfig from the parsed default.yaml params block."""
    cfg = PipelineConfig(intrinsics_path=calib_path)

    raw_dicts = params.get("dictionaries", ["DICT_4X4_50:0.135"])
    cfg.dictionaries = []
    for entry in raw_dicts:
        name, size = entry.split(":")
        cfg.dictionaries.append(DictionaryConfig(name=name.strip(), marker_size_m=float(size)))

    tmo = params.get("T_map_from_opti", {})
    cfg.T_map_from_opti = StaticTransform6DoF(
        x=tmo.get("x", 0.0),    y=tmo.get("y", 0.0),    z=tmo.get("z", 0.0),
        roll=tmo.get("roll", 0.0), pitch=tmo.get("pitch", 0.0), yaw=tmo.get("yaw", 0.0),
    )
    tdc = params.get("T_drone_from_cam", {})
    cfg.T_drone_from_cam = StaticTransform6DoF(
        x=tdc.get("x", 0.0),    y=tdc.get("y", 0.0),    z=tdc.get("z", 0.0),
        roll=tdc.get("roll", 0.0), pitch=tdc.get("pitch", 0.0), yaw=tdc.get("yaw", 0.0),
    )

    opti = params.get("optitrack", {})
    cfg.optitrack_axis = OptiTrackAxisConfig(
        yaw_axis=opti.get("yaw_axis", "z"),
        x_dir=int(opti.get("x_dir", 1)),
        y_dir=int(opti.get("y_dir", 1)),
    )

    agg = params.get("aggregation", {})
    cfg.aggregation = AggregationConfig(
        mad_k=float(agg.get("mad_k", 3.5)),
        min_observations=int(agg.get("min_observations", 2)),
        max_iterations=int(agg.get("max_iterations", 100)),
        convergence_eps=float(agg.get("convergence_eps", 1e-5)),
    )

    qual = params.get("quality", {})
    cfg.quality = QualityConfig(
        blur_thresh=float(qual.get("blur_thresh", 60.0)),
        artifact_thresh=float(qual.get("artifact_thresh", 2.0)),
    )

    proc = params.get("processing", {})
    cfg.max_workers              = int(proc.get("max_workers", 4))
    cfg.frame_stride             = stride
    cfg.max_drone_velocity_m_s   = float(proc.get("max_drone_velocity_m_s", 0.0))
    cfg.max_reproj_err_px        = float(params.get("max_reproj_err_px", 4.0))
    cfg.max_obs_per_marker       = int(params.get("max_obs_per_marker", 200))

    grid  = params.get("grid", {})
    res   = float(grid.get("resolution_m_per_cell", 0.05))
    arena = params.get("arena", {})
    cfg.resolution_m_per_cell = res
    cfg.grid_width_cells  = int(arena.get("width_m",  3.85) / res)
    cfg.grid_height_cells = int(arena.get("height_m", 3.85) / res)

    cfg.verbose = True
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Approach A — stitching
# ─────────────────────────────────────────────────────────────────────────────

def run_stitching(
    video_path:     str,
    csv_path:       str,
    fx: float, fy: float, cx_px: float, cy_px: float,
    T_map_from_opti: np.ndarray,
    dict_name:       str   = "DICT_4X4_50",
    min_marker_px:   float = 20.0,
    frame_stride:    int   = 1,
) -> Dict[int, np.ndarray]:
    """
    Stitching approach — returns {marker_id: (x, y)} in map frame [metres].

    Per-frame:
      1. Detect ArUco in the raw frame (no distortion correction).
      2. Compute pixel offset from principal point to marker centre.
      3. Deproject to metres:  dx = du * altitude / fx
                               dy = -dv * altitude / fy   (y-axis flip: roll=180)
      4. Rotate by drone yaw (from quaternion) to align with OptiTrack frame.
      5. Add to drone (x, y) → marker position in OptiTrack frame.
      6. Apply T_map_from_opti → map frame.

    Aggregation: per-axis median of all accepted observations.
    """
    drone_poses = load_optitrack_csv(csv_path)

    aruco_dict_id = getattr(cv2.aruco, dict_name.strip().upper())
    aruco_dict    = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        def _detect(gray):
            return detector.detectMarkers(gray)
    else:
        def _detect(gray):
            return cv2.aruco.detectMarkers(gray, aruco_dict)

    observations: Dict[int, List[np.ndarray]] = defaultdict(list)
    n_size_dropped = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path!r}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(n_total):
        ok, frame = cap.read()
        if not ok:
            break
        if i >= len(drone_poses):
            break
        if frame_stride > 1 and i % frame_stride != 0:
            continue

        dp       = drone_poses[i]
        altitude = float(dp.pos_xyz[2])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = _detect(gray)
        if ids is None or len(ids) == 0:
            continue

        _, _, yaw = R_to_euler_zyx(dp.R_body)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        for j, marker_id in enumerate(ids.flatten()):
            pts = corners[j].reshape(4, 2)

            # size gate: largest bounding-box dimension in pixels
            side_px = max(pts[:, 0].max() - pts[:, 0].min(),
                          pts[:, 1].max() - pts[:, 1].min())
            if side_px < min_marker_px:
                n_size_dropped += 1
                continue

            # marker centre in image pixels
            mc_u = float(pts[:, 0].mean())
            mc_v = float(pts[:, 1].mean())

            # pixel offset from calibrated principal point
            du = mc_u - cx_px
            dv = mc_v - cy_px

            # deproject to metres using drone altitude and focal length.
            # The camera is mounted with roll=180 (points down), so the
            # image y-axis is inverted relative to the drone body frame.
            dx_body =  du * altitude / fx
            dy_body = -dv * altitude / fy

            # rotate to OptiTrack frame by drone yaw
            dx_opti = cos_y * dx_body - sin_y * dy_body
            dy_opti = sin_y * dx_body + cos_y * dy_body

            # marker position in OptiTrack frame (z=0: markers are on floor)
            p_opti = np.array([
                dp.pos_xyz[0] + dx_opti,
                dp.pos_xyz[1] + dy_opti,
                0.0, 1.0,
            ])

            # map frame
            p_map = T_map_from_opti @ p_opti
            observations[int(marker_id)].append(p_map[:2].copy())

    cap.release()

    if n_size_dropped:
        print(f"  [stitching] {n_size_dropped} detections dropped (< {min_marker_px:.0f} px)")

    # per-axis median
    results: Dict[int, np.ndarray] = {}
    for mid, pts in observations.items():
        arr = np.stack(pts, axis=0)
        results[mid] = np.median(arr, axis=0)
        print(f"  [stitching] id={mid}  n={len(pts)}  "
              f"x={results[mid][0]:+.4f}  y={results[mid][1]:+.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video",  default=os.path.join(_SCAN, "scan.mp4"),
                    help="Path to .mp4 video")
    ap.add_argument("--csv",    default=os.path.join(_SCAN, "telemetry.csv"),
                    help="Path to OptiTrack telemetry CSV")
    ap.add_argument("--calib",  default=_CALIB,
                    help="Path to calibration.yaml (OpenCV format)")
    ap.add_argument("--config", default=_CFG,
                    help="Path to default.yaml (marker_localizer params)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Process every Nth frame (applied to both approaches)")
    ap.add_argument("--min-px", type=float, default=20.0,
                    help="Stitching: minimum marker bounding-box side in pixels")
    ap.add_argument("--gt", default=_GT,
                    help="Path to ground-truth YAML (aruco_pose_gt format); "
                         "pass empty string to skip")
    args = ap.parse_args()

    print(f"Video  : {args.video}")
    print(f"CSV    : {args.csv}")
    print()

    # ── shared setup ─────────────────────────────────────────────────────
    fx, fy, cx_px, cy_px = _load_K(args.calib)
    print(f"Intrinsics  fx={fx:.1f}  fy={fy:.1f}  cx={cx_px:.1f}  cy={cy_px:.1f}")
    print()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    params = raw_cfg["marker_localizer_service"]["ros__parameters"]

    T_map_from_opti = _tmo_from_yaml(params)

    # ── approach A: stitching ─────────────────────────────────────────────
    print("=" * 62)
    print("APPROACH A — Stitching (2D pixel, yaw-only, median)")
    print("=" * 62)
    stitch = run_stitching(
        args.video, args.csv,
        fx, fy, cx_px, cy_px,
        T_map_from_opti,
        frame_stride=args.stride,
        min_marker_px=args.min_px,
    )
    print()

    # ── approach B: full pipeline ─────────────────────────────────────────
    print("=" * 62)
    print("APPROACH B — Full pipeline (solvePnP, 6-DoF chain, MAD+Weiszfeld)")
    print("=" * 62)
    pipeline_cfg = _cfg_from_yaml(args.calib, params, args.stride)
    full, _ = run_pipeline(args.video, args.csv, pipeline_cfg)
    print()

    # ── load ground truth (optional) ─────────────────────────────────────
    gt: Dict[int, np.ndarray] = {}
    if args.gt:
        with open(args.gt) as f:
            gt_raw = yaml.safe_load(f)
        for mid, v in (gt_raw.get("markers") or {}).items():
            gt[int(mid)] = np.array([float(v["x"]), float(v["y"])])
        print(f"Ground truth loaded from {args.gt}  ({len(gt)} marker(s): {sorted(gt)})")
        print()

    # ── side-by-side table ────────────────────────────────────────────────
    all_ids = sorted(set(stitch) | set(full))
    if not all_ids:
        print("No markers found by either approach.")
        return

    have_gt = bool(gt)
    gt_cols = "  {'gt_err_s':>9}  {'gt_err_f':>9}" if have_gt else ""

    print("=" * (82 if have_gt else 62))
    print("COMPARISON" + ("   (gt_err = distance from ground truth)" if have_gt else ""))
    print("=" * (82 if have_gt else 62))
    hdr = (f"  {'ID':>3}  {'stch_x':>8}  {'stch_y':>8}  "
           f"{'full_x':>8}  {'full_y':>8}  {'Δx':>8}  {'Δy':>8}  {'|Δ|m':>7}")
    if have_gt:
        hdr += f"  {'gt_x':>7}  {'gt_y':>7}  {'err_S':>7}  {'err_F':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for mid in all_ids:
        s = stitch.get(mid)
        r = full.get(mid)
        g = gt.get(mid)

        sx   = f"{s[0]:+.4f}" if s is not None else "   N/A  "
        sy   = f"{s[1]:+.4f}" if s is not None else "   N/A  "
        fx_s = f"{r.position_m[0]:+.4f}" if r is not None else "   N/A  "
        fy_s = f"{r.position_m[1]:+.4f}" if r is not None else "   N/A  "

        if s is not None and r is not None:
            dx     = s[0] - r.position_m[0]
            dy     = s[1] - r.position_m[1]
            dist_s = f"{math.hypot(dx, dy):.4f}"
            dx_s   = f"{dx:+.4f}"
            dy_s   = f"{dy:+.4f}"
        else:
            dx_s = dy_s = dist_s = "   N/A  "

        row = (f"  {mid:3d}  {sx:>8}  {sy:>8}  "
               f"{fx_s:>8}  {fy_s:>8}  {dx_s:>8}  {dy_s:>8}  {dist_s:>7}")

        if have_gt:
            gx_s = f"{g[0]:+.4f}" if g is not None else "  N/A  "
            gy_s = f"{g[1]:+.4f}" if g is not None else "  N/A  "
            es_s = f"{math.hypot(s[0]-g[0], s[1]-g[1]):.4f}" \
                   if (s is not None and g is not None) else "  N/A  "
            ef_s = f"{math.hypot(r.position_m[0]-g[0], r.position_m[1]-g[1]):.4f}" \
                   if (r is not None and g is not None) else "  N/A  "
            row += f"  {gx_s:>7}  {gy_s:>7}  {es_s:>7}  {ef_s:>7}"

        print(row)

    print()

    # ── summary stats ─────────────────────────────────────────────────────
    both = [(mid, stitch[mid], full[mid].position_m)
            for mid in all_ids if mid in stitch and mid in full]
    if both:
        dists = [math.hypot(s[0] - f[0], s[1] - f[1]) for _, s, f in both]
        print(f"  Between approaches  — mean |Δ| over {len(both)} marker(s): "
              f"{sum(dists)/len(dists):.4f} m   max: {max(dists):.4f} m")

    if have_gt:
        gt_s = [(mid, stitch[mid], gt[mid]) for mid in all_ids
                if mid in stitch and mid in gt]
        gt_f = [(mid, full[mid].position_m, gt[mid]) for mid in all_ids
                if mid in full and mid in gt]
        if gt_s:
            ds = [math.hypot(s[0]-g[0], s[1]-g[1]) for _, s, g in gt_s]
            print(f"  Stitching  vs GT    — mean err over {len(gt_s)} marker(s): "
                  f"{sum(ds)/len(ds):.4f} m   max: {max(ds):.4f} m")
        if gt_f:
            df = [math.hypot(f[0]-g[0], f[1]-g[1]) for _, f, g in gt_f]
            print(f"  Full pipeline vs GT — mean err over {len(gt_f)} marker(s): "
                  f"{sum(df)/len(df):.4f} m   max: {max(df):.4f} m")


if __name__ == "__main__":
    main()
