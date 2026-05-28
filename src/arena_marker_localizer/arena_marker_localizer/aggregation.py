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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import math
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


def _mad(values: np.ndarray) -> Tuple[float, float]:
    """Median + median-absolute-deviation. Returns (median, MAD)."""
    m = float(np.median(values))
    return m, float(np.median(np.abs(values - m)))


def _circular_median_angle(angles: np.ndarray) -> float:
    """Robust circular 'median' via the angle of the unit-vector median.
    Not the true circular median in the formal sense, but is consistent
    with the per-axis MAD-on-yaw used in the gating step."""
    if len(angles) == 0:
        return 0.0
    cs = np.cos(angles)
    ss = np.sin(angles)
    return float(math.atan2(np.median(ss), np.median(cs)))


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
    positions: np.ndarray,    # (N, 3)
    yaws_rad:  np.ndarray,    # (N,)
    cfg:       Optional[AggregationConfig] = None,
) -> AggregatedPose:
    """Run the full MAD-gate + geometric-median aggregation."""
    cfg = cfg or AggregationConfig()
    positions = np.asarray(positions, dtype=np.float64)
    yaws_rad  = np.asarray(yaws_rad,  dtype=np.float64)
    n_in = len(positions)
    if n_in == 0:
        return AggregatedPose(
            position_m=np.zeros(3), yaw_rad=0.0,
            n_observations=0, rejected=True,
        )

    # Per-axis MAD on x, y, z.
    keep = np.ones(n_in, dtype=bool)
    for i in range(3):
        m, mad = _mad(positions[:, i])
        if mad < 1e-9:
            continue
        deviation = np.abs(positions[:, i] - m)
        keep &= deviation <= cfg.mad_k * mad

    # Per-axis MAD on yaw — but yaw is angular, so compute residuals
    # against the circular median first.
    yaw_ref = _circular_median_angle(yaws_rad)
    yaw_res = np.abs(_angle_residuals(yaws_rad, yaw_ref))
    mad_y = float(np.median(yaw_res))
    if mad_y > 1e-9:
        keep &= yaw_res <= cfg.mad_k * mad_y

    if keep.sum() < cfg.min_observations:
        return AggregatedPose(
            position_m=np.zeros(3), yaw_rad=0.0,
            n_observations=int(keep.sum()), rejected=True,
        )

    surviving = positions[keep]
    surviving_y = yaws_rad[keep]

    pos = _geometric_median(surviving, cfg)
    yaw = _circular_median_angle(surviving_y)


    
    return AggregatedPose(
        position_m=pos, yaw_rad=yaw,
        n_observations=int(keep.sum()), rejected=False,
    )
