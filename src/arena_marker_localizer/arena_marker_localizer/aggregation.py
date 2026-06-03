"""
arena_marker_localizer.aggregation
─────────────────────────────────────────────────────────────────────────────
Combine many per-frame observations of the same marker into one
aggregate pose, robust to outliers.

Two stages:
  1. Per-axis MAD gate on (x, y, z, yaw):
        |value - median| > k * MAD  →  drop the whole observation.
     A single bad axis kicks the entire observation out (so the angle
     and the position are estimated from the same set of survivors).
  2. Geometric median of the surviving 3D positions (Weiszfeld
     iteration). Yaw is aggregated separately as the circular median
     of the survivors (sin/cos averaging).

The geometric median minimises the sum of L2 distances; it's robust to a
substantial fraction of outliers (breakdown point ~50%) and is better
behaved than the per-axis median when the noise is anisotropic.

Drone attitude fields (mean_obs_drone_roll/pitch/yaw_rad)
─────────────────────────────────────────────────────────
All three angles are stored in the MAP frame, as extracted from
R_map_from_drone = R_map_from_opti @ R_opti_from_drone(t).
This means calibrate_bias can reconstruct R_map_from_drone directly via
euler_zyx_to_R(roll, pitch, yaw) without needing T_map_from_opti again.
When the CSV lacks roll/pitch columns (legacy mode) these default to 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AggregationConfig:
    mad_k: float = 3.5
    """Per-axis MAD gate factor. Standard choice for 'robust 3-sigma'."""
    min_observations: int = 2
    """If fewer than this many survive the gate, the marker is rejected."""
    max_iterations: int = 100
    """Weiszfeld iterations cap."""
    convergence_eps: float = 1e-5
    """Stop Weiszfeld when the position update is below this (metres)."""


@dataclass
class AggregatedPose:
    position_m:      np.ndarray   # (3,)
    yaw_rad:         float
    n_observations:  int
    rejected:        bool         # True if min_observations not met
    pos_var_x:      float = 1.0   # sample variance of inlier x positions [m²]
    pos_var_y:      float = 1.0   # sample variance of inlier y positions [m²]
    pos_cov_xy:     float = 0.0   # sample covariance of inlier (x,y) [m²]
    yaw_var:        float = 1.0   # -2·ln(R_mean), circular dispersion [rad²]
    # ── Drone attitude in MAP frame across surviving observations ──────
    # Used by calibrate_bias to build the full rotation design matrix.
    # Circular mean via (sin, cos) averaging over inlier observations.
    # All zero when the CSV lacks roll/pitch columns (legacy mode).
    mean_obs_drone_roll_rad:  float = 0.0
    mean_obs_drone_pitch_rad: float = 0.0
    mean_obs_drone_yaw_rad:   float = 0.0


def _mad(values: np.ndarray) -> Tuple[float, float]:
    """Median + median-absolute-deviation. Returns (median, MAD)."""
    m = float(np.median(values))
    return m, float(np.median(np.abs(values - m)))


def _circular_mean_angle(angles: np.ndarray) -> float:
    """Circular mean via unit-vector averaging."""
    if len(angles) == 0:
        return 0.0
    return float(math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _circular_median_angle(angles: np.ndarray) -> float:
    """Robust circular 'median' via the angle of the unit-vector median."""
    if len(angles) == 0:
        return 0.0
    return float(math.atan2(np.median(np.sin(angles)), np.median(np.cos(angles))))


def _wrap_angle_signed(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _angle_residuals(angles: np.ndarray, ref: float) -> np.ndarray:
    """Smallest signed distance from each angle to ref, in [-pi, pi]."""
    return np.array([_wrap_angle_signed(a - ref) for a in angles])


def _geometric_median(points: np.ndarray, cfg: AggregationConfig) -> np.ndarray:
    """Weiszfeld iteration for the geometric median in R^d."""
    if len(points) == 1:
        return points[0].copy()

    x = points.mean(axis=0)
    for _ in range(cfg.max_iterations):
        diffs = points - x
        dists = np.linalg.norm(diffs, axis=1)
        nonzero = dists > 1e-9
        if not nonzero.any():
            break
        weights = 1.0 / dists[nonzero]
        x_new = (points[nonzero] * weights[:, None]).sum(axis=0) \
                / weights.sum()
        if np.linalg.norm(x_new - x) < cfg.convergence_eps:
            x = x_new
            break
        x = x_new
    return x


def aggregate(
    positions: np.ndarray,                             # (N, 3)
    yaws_rad:  np.ndarray,                             # (N,)
    cfg: Optional[AggregationConfig] = None,
    drone_yaws_rad:   Optional[np.ndarray] = None,     # (N,) map-frame yaw
    drone_rolls_rad:  Optional[np.ndarray] = None,     # (N,) map-frame roll
    drone_pitches_rad: Optional[np.ndarray] = None,    # (N,) map-frame pitch
) -> AggregatedPose:
    """Aggregate N per-frame observations into one robust pose estimate.

    drone_rolls/pitches/yaws_rad should be the drone attitude in the MAP
    frame for each observation (extracted from R_map_from_drone).  Only
    the surviving-inlier subset is averaged.
    """
    cfg       = cfg or AggregationConfig()
    positions = np.asarray(positions, dtype=np.float64)
    yaws_rad  = np.asarray(yaws_rad,  dtype=np.float64)
    n_in      = len(positions)

    if n_in == 0:
        return AggregatedPose(
            position_m=np.zeros(3), yaw_rad=0.0,
            n_observations=0, rejected=True,
        )

    # ── MAD gate: x, y, z ────────────────────────────────────────────────
    keep = np.ones(n_in, dtype=bool)
    for i in range(3):
        m, mad = _mad(positions[:, i])
        if mad < 1e-9:
            continue
        keep &= np.abs(positions[:, i] - m) <= cfg.mad_k * mad

    # ── MAD gate: yaw (circular) ─────────────────────────────────────────
    yaw_ref = _circular_median_angle(yaws_rad)
    yaw_res = np.abs(_angle_residuals(yaws_rad, yaw_ref))
    mad_y   = float(np.median(yaw_res))
    if mad_y > 1e-9:
        keep &= yaw_res <= cfg.mad_k * mad_y

    if keep.sum() < cfg.min_observations:
        return AggregatedPose(
            position_m=np.zeros(3), yaw_rad=0.0,
            n_observations=int(keep.sum()), rejected=True,
        )

    surviving   = positions[keep]    # (M, 3)
    surviving_y = yaws_rad[keep]     # (M,)

    # ── Drone attitude in MAP frame (circular mean over inliers) ─────────
    def _circ_mean_inliers(arr: Optional[np.ndarray]) -> float:
        if arr is None:
            return 0.0
        sub = np.asarray(arr, dtype=np.float64)[keep]
        return _circular_mean_angle(sub)

    mean_drone_yaw   = _circ_mean_inliers(drone_yaws_rad)
    mean_drone_roll  = _circ_mean_inliers(drone_rolls_rad)
    mean_drone_pitch = _circ_mean_inliers(drone_pitches_rad)

    # ── Robust pose estimate ─────────────────────────────────────────────
    pos = _geometric_median(surviving, cfg)
    yaw = _circular_median_angle(surviving_y)

    # ── Position covariance: residuals from the geometric median ─────────
    if len(surviving) > 1:
        res = surviving - pos   # (M, 3)
        dof = len(surviving) - 1
        pos_var_x  = float(np.dot(res[:, 0], res[:, 0]) / dof)
        pos_var_y  = float(np.dot(res[:, 1], res[:, 1]) / dof)
        pos_cov_xy = float(np.dot(res[:, 0], res[:, 1]) / dof)
    else:
        pos_var_x  = 1.0
        pos_var_y  = 1.0
        pos_cov_xy = 0.0

    # ── Circular dispersion of yaw [rad²] ────────────────────────────────
    R_mean  = math.sqrt(np.mean(np.sin(surviving_y))**2 +
                        np.mean(np.cos(surviving_y))**2)
    R_mean  = min(R_mean, 1.0 - 1e-9)   # guard log(0)
    yaw_var = float(-2.0 * math.log(R_mean))

    return AggregatedPose(
        position_m=pos,
        yaw_rad=yaw,
        n_observations=int(keep.sum()),
        rejected=False,
        pos_var_x=pos_var_x,
        pos_var_y=pos_var_y,
        pos_cov_xy=pos_cov_xy,
        yaw_var=yaw_var,
        mean_obs_drone_roll_rad=mean_drone_roll,
        mean_obs_drone_pitch_rad=mean_drone_pitch,
        mean_obs_drone_yaw_rad=mean_drone_yaw,
    )
