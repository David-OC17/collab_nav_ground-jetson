"""
arena_marker_localizer.pipeline
─────────────────────────────────────────────────────────────────────────────
End-to-end Python pipeline:

    read video frame N
    read CSV row    N
    quality filter   (drop noisy / blurry / artifact-heavy frames)
    detect markers (multi-dictionary ArUco)
    per-marker solvePnP -> T_cam_from_marker
    full static chain -> marker pose in MAP frame
    accumulate per-marker observations (cap at max_obs_per_marker)
    aggregate per marker (MAD gate + geometric median)
    return {marker_id: AggregatedPose} plus cell-index of each pose
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import cv2

from .intrinsics       import CameraIntrinsics, load_intrinsics
from .quality          import QualityConfig, frame_passes
from .optitrack        import DronePose, load_optitrack_csv
from .transforms       import (
    StaticTransform6DoF, OptiTrackAxisConfig,
    opti_transform_from_pose, marker_in_map,
    R_to_euler_zyx,
)
from .marker_detection import (
    MultiDictDetector, DictionaryConfig, Detection,
)
from .aggregation      import AggregationConfig, AggregatedPose, aggregate


# ─────────────────────────────────────────────────────────────────────────
# Top-level configuration
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Intrinsics ─────────────────────────────────────────────────────
    intrinsics_path: str = ""
    intrinsics_camera_matrix_key: str = "camera_matrix"
    intrinsics_dist_key:          str = "dist_coeff"

    # ── ArUco dictionaries ────────────────────────────────────────────
    dictionaries: List[DictionaryConfig] = field(default_factory=lambda: [
        DictionaryConfig(name="DICT_4X4_50", marker_size_m=0.10),
    ])

    # ── Quality filter ────────────────────────────────────────────────
    quality: QualityConfig = field(default_factory=QualityConfig)

    # ── Static transform chain (all configurable) ─────────────────────
    T_drone_from_cam: StaticTransform6DoF = field(
        default_factory=StaticTransform6DoF
    )
    """Camera mounting on the drone. Includes both the camera position
    on the body and the rotation from OpenCV's camera frame
    (X right, Y down, Z forward) to your drone-body frame.
    Default = identity (override per setup)."""

    T_map_from_opti: StaticTransform6DoF = field(
        default_factory=StaticTransform6DoF
    )
    """Map-from-OptiTrack: where the arena origin (bottom-left) sits in
    the OptiTrack frame, plus the rotation to align +X right / +Y up
    with the map's nav2 convention. Default = identity (override)."""

    optitrack_axis: OptiTrackAxisConfig = field(
        default_factory=OptiTrackAxisConfig
    )

    # ── Detection-quality gates ───────────────────────────────────────
    max_reproj_err_px: float = 4.0
    """Drop a single detection whose solvePnP reprojection error is
    above this. Cheap noise filter before aggregation."""

    # ── Aggregation ────────────────────────────────────────────────────
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    max_obs_per_marker: int = 200
    """Cap on observations stored per marker. When exceeded the
    oldest observations are dropped (FIFO)."""

    # ── Map → grid cell conversion ────────────────────────────────────
    resolution_m_per_cell: float = 0.05
    grid_width_cells:  int = 80
    grid_height_cells: int = 80

    # ── Parallel processing ───────────────────────────────────────────
    max_workers: int = 4
    """Number of parallel threads for frame processing. Each thread
    holds its own MultiDictDetector instance. Set to 1 to disable
    parallelism (useful for debugging)."""

    frame_stride: int = 1
    """Process every Nth video frame. 1 = every frame. 2 = every other
    frame (halves the workload at 30 fps, still yields ample observations
    for aggregation at min_observations=50)."""

    # ── Velocity gate ─────────────────────────────────────────────────
    max_drone_velocity_m_s: float = 0.0
    """Drop frames where the drone speed exceeds this threshold [m/s].
    0.0 = disabled (keep all frames).
    When the drone is translating it maintains a pitch/roll proportional
    to speed; dropping fast frames reduces the systematic position error
    that results from ignoring pitch/roll in the transform chain.
    Recommended starting value: 0.15 m/s."""

    # ── Diagnostics ────────────────────────────────────────────────────
    verbose: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Per-frame observation container
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _MarkerObservations:
    positions: List[np.ndarray] = field(default_factory=list)
    yaws_rad:  List[float]      = field(default_factory=list)
    sample_label: List[str]     = field(default_factory=list)
    """e.g. 'DICT_4X4_50' so the caller can audit which dict produced
    the aggregate."""

    def add(self, pos: np.ndarray, yaw: float, label: str, cap: int):
        self.positions.append(pos)
        self.yaws_rad.append(yaw)
        self.sample_label.append(label)
        # FIFO trim
        if len(self.positions) > cap:
            self.positions  = self.positions[-cap:]
            self.yaws_rad   = self.yaws_rad[-cap:]
            self.sample_label = self.sample_label[-cap:]


# ─────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MarkerResult:
    marker_id:      int
    dict_name:      str
    position_m:     np.ndarray   # (3,) in map frame
    yaw_rad:        float
    cell_x:         int
    cell_y:         int
    n_observations: int
    pos_var_x:     float = 1.0  # ← Sample variance of inlier x positions [m²]
    pos_var_y:     float = 1.0  # ← Sample variance of inlier y positions [m²]
    pos_cov_xy:    float = 0.0  # ← Sample covariance of inlier (x,y) [m²]
    yaw_var:       float = 1.0  # ← Circular dispersion of inlier yaws [rad²]


# ─────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ─────────────────────────────────────────────────────────────────────────

def _xyz_to_cell(pos: np.ndarray, cfg: PipelineConfig) -> Tuple[int, int]:
    """Convert (x,y,z) in metres into the OccupancyGrid cell index.
    Origin is the bottom-left of the arena bbox (nav2 standard); +x right,
    +y up. Clamped to grid bounds."""
    cx = int(pos[0] / cfg.resolution_m_per_cell)
    cy = int(pos[1] / cfg.resolution_m_per_cell)
    cx = max(0, min(cfg.grid_width_cells  - 1, cx))
    cy = max(0, min(cfg.grid_height_cells - 1, cy))
    return cx, cy


def run_pipeline(
    video_path:    str,
    csv_path:      str,
    cfg:           PipelineConfig,
) -> Tuple[Dict[int, MarkerResult], List[DronePose]]:
    """Run the full pipeline.

    Returns
    ───────
    results       : {marker_id: MarkerResult} for each accepted marker.
                    Markers rejected during aggregation are omitted.
    drone_poses   : the full OptiTrack trajectory loaded from the CSV
                    (one DronePose per CSV row), so downstream debug /
                    visualization code doesn't need to re-read the file.
    """
    # ── 1) Load intrinsics, CSV, and pre-build the static transforms ──
    if not cfg.intrinsics_path:
        raise ValueError("PipelineConfig.intrinsics_path is required.")
    intrinsics = load_intrinsics(
        cfg.intrinsics_path,
        matrix_key=cfg.intrinsics_camera_matrix_key,
        dist_key=cfg.intrinsics_dist_key,
    )

    drone_poses = load_optitrack_csv(csv_path)
    if not drone_poses:
        raise ValueError(f"OptiTrack CSV {csv_path!r} is empty.")

    T_drone_from_cam = cfg.T_drone_from_cam.as_matrix()
    T_map_from_opti  = cfg.T_map_from_opti.as_matrix()

    # Apply the configurable x_dir / y_dir flip as part of T_map_from_opti.
    # We multiply the rotation by a diagonal sign matrix on the right.
    flip = np.diag([cfg.optitrack_axis.x_dir,
                    cfg.optitrack_axis.y_dir,
                    1, 1]).astype(np.float64)
    T_map_from_opti = T_map_from_opti @ flip

    # ── 2) Open the video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path!r}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Thread-local detector: one MultiDictDetector per worker thread ──
    # ArucoDetector has internal mutable buffers; sharing across threads
    # is unsafe. Each thread creates its own instance on first use.
    _tl = threading.local()

    def _get_detector() -> MultiDictDetector:
        if not hasattr(_tl, "det"):
            _tl.det = MultiDictDetector(cfg.dictionaries)
        return _tl.det

    def _process_frame(frame: np.ndarray, dp: DronePose) -> dict:
        passed, _blur, _art = frame_passes(frame, cfg.quality)
        if not passed:
            return {"kind": "quality_fail"}

        detections = _get_detector().detect(frame, intrinsics)
        if not detections:
            return {"kind": "no_marker"}

        T_opti_drone = opti_transform_from_pose(
            dp.pos_xyz, dp.yaw_rad, cfg.optitrack_axis,
        )
        obs_list = []
        n_pnp_fail = 0
        for det in detections:
            if det.reproj_err > cfg.max_reproj_err_px:
                n_pnp_fail += 1
                continue
            T_map_marker = marker_in_map(
                det.T_cam_marker, T_opti_drone,
                T_drone_from_cam, T_map_from_opti,
            )
            pos = T_map_marker[:3, 3].copy()
            _r, _p, yaw = R_to_euler_zyx(T_map_marker[:3, :3])
            obs_list.append((det.marker_id, pos, yaw, det.dict_name))

        if obs_list:
            return {"kind": "detected", "obs": obs_list, "pnp_fail": n_pnp_fail}
        return {"kind": "no_marker", "pnp_fail": n_pnp_fail}

    # ── Precompute per-frame drone speed ─────────────────────────────
    # Central differences for interior frames; forward/backward at edges.
    # Speed is in m/s using OptiTrack timestamps for Δt.
    n_poses = len(drone_poses)
    drone_speed_m_s: List[float] = [0.0] * n_poses
    for _i in range(n_poses):
        _i0 = max(0, _i - 1)
        _i1 = min(n_poses - 1, _i + 1)
        dt = drone_poses[_i1].timestamp_sec - drone_poses[_i0].timestamp_sec
        if dt > 1e-9:
            drone_speed_m_s[_i] = float(
                np.linalg.norm(drone_poses[_i1].pos_xyz - drone_poses[_i0].pos_xyz) / dt
            )

    vel_thresh = cfg.max_drone_velocity_m_s

    observations: Dict[int, _MarkerObservations] = {}
    dict_name_by_id: Dict[int, str] = {}
    stats = dict(read=0, quality_fail=0, no_marker=0, accepted=0,
                 csv_short=0, pnp_fail=0, vel_filtered=0)

    n_workers = max(1, cfg.max_workers)
    max_inflight = n_workers * 2     # bound frames held in memory
    stride = max(1, cfg.frame_stride)
    pending: List[Future] = []

    def _drain_one() -> None:
        result = pending.pop(0).result()
        kind = result["kind"]
        if kind == "quality_fail":
            stats["quality_fail"] += 1
        elif kind == "no_marker":
            stats["no_marker"] += 1
            stats["pnp_fail"] += result.get("pnp_fail", 0)
        else:
            stats["pnp_fail"] += result.get("pnp_fail", 0)
            for marker_id, pos, yaw, dict_name in result["obs"]:
                obs = observations.setdefault(marker_id, _MarkerObservations())
                obs.add(pos, yaw, dict_name, cfg.max_obs_per_marker)
                dict_name_by_id[marker_id] = dict_name
                stats["accepted"] += 1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i in range(n_total):
            ok, frame = cap.read()
            if not ok:
                break
            stats["read"] += 1

            if i >= len(drone_poses):
                stats["csv_short"] += 1
                continue

            if stride > 1 and i % stride != 0:
                continue

            if vel_thresh > 0.0 and drone_speed_m_s[i] > vel_thresh:
                stats["vel_filtered"] += 1
                continue

            while len(pending) >= max_inflight:
                _drain_one()

            pending.append(pool.submit(_process_frame, frame, drone_poses[i]))

        while pending:
            _drain_one()

    cap.release()

    if cfg.verbose:
        pct_vel = (100.0 * stats["vel_filtered"] / max(1, stats["read"]))
        print(f"  Pipeline stats: {stats}  ({pct_vel:.1f}% dropped by velocity gate)")
        print(f"  Unique marker IDs observed: {sorted(observations.keys())}")

    # ── 3) Aggregate per marker ───────────────────────────────────────
    results: Dict[int, MarkerResult] = {}
    for marker_id, obs in observations.items():
        positions = np.stack(obs.positions, axis=0)
        yaws_rad  = np.array(obs.yaws_rad,  dtype=np.float64)
        agg = aggregate(positions, yaws_rad, cfg.aggregation)
        if agg.rejected:
            if cfg.verbose:
                print(f"    [reject] id={marker_id} survivors="
                      f"{agg.n_observations}")
            continue
        cell_x, cell_y = _xyz_to_cell(agg.position_m, cfg)
        results[marker_id] = MarkerResult(
            marker_id=marker_id,
            dict_name=dict_name_by_id.get(marker_id, ""),
            position_m=agg.position_m,
            yaw_rad=agg.yaw_rad,
            cell_x=cell_x,
            cell_y=cell_y,
            n_observations=agg.n_observations,
            pos_var_x=agg.pos_var_x,
            pos_var_y=agg.pos_var_y,
            pos_cov_xy=agg.pos_cov_xy,
            yaw_var=agg.yaw_var,
        )

    return results, drone_poses