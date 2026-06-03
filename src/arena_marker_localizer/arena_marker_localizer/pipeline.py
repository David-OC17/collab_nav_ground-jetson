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
    compose_T, marker_in_map,
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
    """Camera mounting on the drone."""

    T_map_from_opti: StaticTransform6DoF = field(
        default_factory=StaticTransform6DoF
    )
    """Map-from-OptiTrack: arena origin (bottom-left) in the OptiTrack
    frame, plus the rotation to align +X right / +Y up."""

    optitrack_axis: OptiTrackAxisConfig = field(
        default_factory=OptiTrackAxisConfig
    )

    # ── Detection-quality gates ───────────────────────────────────────
    max_reproj_err_px: float = 4.0

    # ── Aggregation ────────────────────────────────────────────────────
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    max_obs_per_marker: int = 200

    # ── Map → grid cell conversion ────────────────────────────────────
    resolution_m_per_cell: float = 0.05
    grid_width_cells:  int = 80
    grid_height_cells: int = 80

    # ── Parallel processing ───────────────────────────────────────────
    max_workers: int = 4
    frame_stride: int = 1

    # ── Velocity gate ─────────────────────────────────────────────────
    max_drone_velocity_m_s: float = 0.0
    """Drop frames where the drone speed exceeds this threshold [m/s].
    0.0 = disabled.  With quaternion CSVs, attitude is always exact;
    this gate is useful mainly for reducing motion-blur observations."""

    # ── Diagnostics ────────────────────────────────────────────────────
    verbose: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Per-frame observation container
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _MarkerObservations:
    positions:        List[np.ndarray] = field(default_factory=list)
    yaws_rad:         List[float]      = field(default_factory=list)
    drone_rolls_rad:  List[float]      = field(default_factory=list)
    drone_pitches_rad: List[float]     = field(default_factory=list)
    drone_yaws_rad:   List[float]      = field(default_factory=list)
    sample_label:     List[str]        = field(default_factory=list)

    def add(self, pos: np.ndarray, yaw: float, label: str, cap: int,
            drone_roll: float = 0.0,
            drone_pitch: float = 0.0,
            drone_yaw: float = 0.0):
        self.positions.append(pos)
        self.yaws_rad.append(yaw)
        self.drone_rolls_rad.append(drone_roll)
        self.drone_pitches_rad.append(drone_pitch)
        self.drone_yaws_rad.append(drone_yaw)
        self.sample_label.append(label)
        # FIFO trim
        if len(self.positions) > cap:
            self.positions         = self.positions[-cap:]
            self.yaws_rad          = self.yaws_rad[-cap:]
            self.drone_rolls_rad   = self.drone_rolls_rad[-cap:]
            self.drone_pitches_rad = self.drone_pitches_rad[-cap:]
            self.drone_yaws_rad    = self.drone_yaws_rad[-cap:]
            self.sample_label      = self.sample_label[-cap:]


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
    pos_var_x:     float = 1.0
    pos_var_y:     float = 1.0
    pos_cov_xy:    float = 0.0
    yaw_var:       float = 1.0
    # ── Drone attitude in MAP frame, circular mean over inlier obs ────
    mean_obs_drone_roll_rad:  float = 0.0
    mean_obs_drone_pitch_rad: float = 0.0
    mean_obs_drone_yaw_rad:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ─────────────────────────────────────────────────────────────────────────

def _xyz_to_cell(pos: np.ndarray, cfg: PipelineConfig) -> Tuple[int, int]:
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
    drone_poses   : the full OptiTrack trajectory from the CSV.
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

    # x_dir / y_dir flip applied as a right-multiplied sign matrix on
    # T_map_from_opti so it only affects the OptiTrack-frame interpretation.
    flip = np.diag([cfg.optitrack_axis.x_dir,
                    cfg.optitrack_axis.y_dir,
                    1, 1]).astype(np.float64)
    T_map_from_opti = T_map_from_opti @ flip

    # ── 2) Open the video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path!r}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        # Drone pose in OptiTrack frame — R_body built from quaternion in
        # optitrack.py; no Euler conversion or singularity risk here.
        T_opti_drone = compose_T(dp.pos_xyz, dp.R_body)

        # Drone attitude in MAP frame — stored per-observation so
        # calibrate_bias can build the full rotation design matrix.
        T_map_drone = T_map_from_opti @ T_opti_drone
        drone_roll_map, drone_pitch_map, drone_yaw_map = \
            R_to_euler_zyx(T_map_drone[:3, :3])

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
            obs_list.append((
                det.marker_id, pos, yaw, det.dict_name,
                drone_roll_map, drone_pitch_map, drone_yaw_map,
            ))

        if obs_list:
            return {"kind": "detected", "obs": obs_list, "pnp_fail": n_pnp_fail}
        return {"kind": "no_marker", "pnp_fail": n_pnp_fail}

    # ── Precompute per-frame drone speed ─────────────────────────────
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

    n_workers  = max(1, cfg.max_workers)
    max_inflight = n_workers * 2
    stride     = max(1, cfg.frame_stride)
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
            for (marker_id, pos, yaw, dict_name,
                 d_roll, d_pitch, d_yaw) in result["obs"]:
                obs = observations.setdefault(marker_id, _MarkerObservations())
                obs.add(pos, yaw, dict_name, cfg.max_obs_per_marker,
                        drone_roll=d_roll,
                        drone_pitch=d_pitch,
                        drone_yaw=d_yaw)
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
        positions         = np.stack(obs.positions, axis=0)
        yaws_rad          = np.array(obs.yaws_rad,          dtype=np.float64)
        drone_rolls_rad   = np.array(obs.drone_rolls_rad,   dtype=np.float64)
        drone_pitches_rad = np.array(obs.drone_pitches_rad, dtype=np.float64)
        drone_yaws_rad    = np.array(obs.drone_yaws_rad,    dtype=np.float64)

        agg = aggregate(
            positions, yaws_rad, cfg.aggregation,
            drone_yaws_rad=drone_yaws_rad,
            drone_rolls_rad=drone_rolls_rad,
            drone_pitches_rad=drone_pitches_rad,
        )
        if agg.rejected:
            if cfg.verbose:
                print(f"    [reject] id={marker_id} survivors={agg.n_observations}")
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
            mean_obs_drone_roll_rad=agg.mean_obs_drone_roll_rad,
            mean_obs_drone_pitch_rad=agg.mean_obs_drone_pitch_rad,
            mean_obs_drone_yaw_rad=agg.mean_obs_drone_yaw_rad,
        )

    return results, drone_poses
