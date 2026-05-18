"""
arena_marker_localizer.transforms
─────────────────────────────────────────────────────────────────────────────
Homogeneous-transform helpers and the static camera → map chain.

Chain
─────
    T_marker_in_map   =   T_map_from_opti
                        @ T_opti_from_drone(t)
                        @ T_drone_from_cam
                        @ T_cam_from_marker

Pieces
──────
  T_cam_from_marker     : per-frame, from solvePnP (the only dynamic part
                          when the camera is fixed to the drone).
  T_drone_from_cam      : static, configurable as 6 numbers (x, y, z,
                          roll, pitch, yaw). Camera mounting on the drone.
  T_opti_from_drone(t)  : per-frame, built from one CSV row
                          (translation + yaw-only rotation).
  T_map_from_opti       : static, configurable as 6 numbers. The arena
                          map's bottom-left corner with optional axis
                          re-mapping (e.g. flip x or y if the OptiTrack
                          axes don't match the map convention).

Conventions
───────────
  - All rotations are intrinsic Tait-Bryan ZYX (yaw, pitch, roll), i.e.
    Rz(yaw) @ Ry(pitch) @ Rx(roll), the standard "yaw about Z" convention.
  - For OptiTrack the yaw axis can be configured to Y if your rig is
    Y-up (this is handled in opti_transform_from_pose).
  - All inputs are radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Static-transform configuration containers
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class StaticTransform6DoF:
    """One static rigid transform, expressed as 6 numbers."""
    x:     float = 0.0
    y:     float = 0.0
    z:     float = 0.0
    roll:  float = 0.0
    pitch: float = 0.0
    yaw:   float = 0.0

    def as_matrix(self) -> np.ndarray:
        return compose_T(np.array([self.x, self.y, self.z]),
                         euler_zyx_to_R(self.roll, self.pitch, self.yaw))


@dataclass
class OptiTrackAxisConfig:
    """How OptiTrack's coordinate frame relates to the map frame.

    `yaw_axis` says about which axis the CSV's `yaw` column rotates:
      "z"  (default) — OptiTrack Z-up: yaw = rotation about Z
      "y"            — Y-up: yaw = rotation about Y

    `x_dir` / `y_dir` let you flip OptiTrack X or Y to align with map
    +X / +Y. Each is +1 or -1. The map origin is the *arena bottom-left*
    (nav2 OccupancyGrid convention); the user-configurable `T_map_from_opti`
    handles the translation from OptiTrack origin (typically arena
    center) to the bottom-left, plus any axis swap if needed.
    """
    yaw_axis: str = "z"   # "z" or "y"
    x_dir:    int = +1    # +1 or -1
    y_dir:    int = +1


# ─────────────────────────────────────────────────────────────────────────
# Primitive geometry helpers
# ─────────────────────────────────────────────────────────────────────────

def euler_zyx_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Tait-Bryan ZYX: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0,  0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0,  1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def R_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of euler_zyx_to_R. Returns (roll, pitch, yaw) in radians."""
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))   # clamp for numerical safety
    pitch = math.asin(sp)
    if abs(sp) > 0.999999:
        # Gimbal lock; yaw is undetermined — assign yaw to 0 by convention.
        roll  = math.atan2(-R[1, 2], R[1, 1])
        yaw   = 0.0
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def R_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (x, y, z, w) quaternion."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def compose_T(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation + rotation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t.reshape(3)
    return T


def decompose_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (translation (3,), rotation (3,3))."""
    return T[:3, 3].copy(), T[:3, :3].copy()


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


# ─────────────────────────────────────────────────────────────────────────
# Per-frame OptiTrack pose -> homogeneous transform
# ─────────────────────────────────────────────────────────────────────────

def opti_transform_from_pose(pos_xyz: np.ndarray,
                             yaw_rad: float,
                             axis_cfg: OptiTrackAxisConfig,
                             ) -> np.ndarray:
    """T_opti_from_drone for one CSV row: rotation about the configured
    yaw axis, plus translation. Pitch/roll are assumed 0 (the CSV
    doesn't carry them; the OptiTrack rig presumably stabilizes the
    drone well enough that we treat the yaw column as the only
    rotational degree of freedom)."""
    if axis_cfg.yaw_axis == "z":
        R = euler_zyx_to_R(0.0, 0.0, yaw_rad)
    elif axis_cfg.yaw_axis == "y":
        # yaw about Y: equivalent to setting pitch in ZYX terms.
        R = euler_zyx_to_R(0.0, yaw_rad, 0.0)
    else:
        raise ValueError(f"Unknown yaw_axis {axis_cfg.yaw_axis!r}; "
                         f"expected 'z' or 'y'.")
    return compose_T(pos_xyz, R)


# ─────────────────────────────────────────────────────────────────────────
# The static OpenCV-camera-to-body-frame rotation
# ─────────────────────────────────────────────────────────────────────────
# OpenCV camera frame: +X right, +Y down, +Z forward (into the scene).
# A typical drone body frame: +X forward, +Y left, +Z up.
# Going from camera to body requires a known fixed rotation, which is
# baked into the user-supplied T_drone_from_cam (StaticTransform6DoF).
# We don't hardcode a value here — let the user set the 6 numbers.


# ─────────────────────────────────────────────────────────────────────────
# Full static chain
# ─────────────────────────────────────────────────────────────────────────

def marker_in_map(
    T_cam_from_marker:  np.ndarray,
    drone_pose_in_opti: np.ndarray,   # T_opti_from_drone (this frame)
    T_drone_from_cam:   np.ndarray,
    T_map_from_opti:    np.ndarray,
) -> np.ndarray:
    """Compose the full chain to get the marker's pose in the map frame."""
    return (T_map_from_opti
            @ drone_pose_in_opti
            @ T_drone_from_cam
            @ T_cam_from_marker)
