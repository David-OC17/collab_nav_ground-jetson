"""
arena_marker_localizer.optitrack
─────────────────────────────────────────────────────────────────────────────
Read OptiTrack drone pose log.

Supported CSV formats (auto-detected by column names):

  Quaternion format (preferred — no Euler singularities):
    timestamp_sec,frame_id,pos_x,pos_y,pos_z,qx,qy,qz,qw

  Full-pose Euler format (legacy):
    timestamp_sec,frame_id,pos_x,pos_y,pos_z,yaw,pitch,roll

  Yaw-only format (oldest legacy):
    timestamp_sec,frame_id,pos_x,pos_y,pos_z,yaw

All formats are converted to a 3×3 rotation matrix R_body (body → OptiTrack)
stored in DronePose.  Downstream code composes T_opti_from_drone = [R_body | pos]
without any further angle arithmetic.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .transforms import euler_zyx_to_R, quaternion_to_R


@dataclass
class DronePose:
    timestamp_sec: float
    frame_id:      int
    pos_xyz:       np.ndarray   # (3,) float64
    R_body:        np.ndarray   # (3,3) rotation matrix: body frame → OptiTrack frame


def load_optitrack_csv(path: str) -> List[DronePose]:
    """Read the CSV; return one DronePose per row in file order.

    Auto-detects format from column names:
      • quaternion (qx,qy,qz,qw) — preferred
      • full-pose Euler (roll, pitch) — legacy
      • yaw-only — oldest legacy; assumes yaw is a Z-axis rotation
    """
    rows: List[DronePose] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        has_quat      = {"qx", "qy", "qz", "qw"}.issubset(fieldnames)
        has_roll      = "roll"  in fieldnames
        has_pitch     = "pitch" in fieldnames
        has_full_euler = has_roll and has_pitch

        if not has_quat and not has_full_euler:
            if not has_roll and not has_pitch:
                warnings.warn(
                    f"CSV {path!r} has only 'yaw' (no quaternion, no roll/pitch). "
                    "Falling back to yaw-only rotation (Z-axis). "
                    "Re-record with record_scan for quaternion output.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"CSV {path!r} has Euler angles but no quaternion columns. "
                    "Consider re-recording with record_scan for quaternion output.",
                    UserWarning,
                    stacklevel=2,
                )

        for row in reader:
            try:
                pos = np.array(
                    [float(row["pos_x"]),
                     float(row["pos_y"]),
                     float(row["pos_z"])],
                    dtype=np.float64,
                )

                if has_quat:
                    qx = float(row["qx"])
                    qy = float(row["qy"])
                    qz = float(row["qz"])
                    qw = float(row["qw"])
                    # Normalise to guard against small float drift in logging
                    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
                    if n > 1e-9:
                        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
                    R = quaternion_to_R(qx, qy, qz, qw)

                elif has_full_euler:
                    R = euler_zyx_to_R(
                        float(row["roll"]),
                        float(row["pitch"]),
                        float(row["yaw"]),
                    )

                else:
                    # Yaw-only legacy: assume yaw rotates about Z
                    R = euler_zyx_to_R(0.0, 0.0, float(row["yaw"]))

                pose = DronePose(
                    timestamp_sec=float(row["timestamp_sec"]),
                    frame_id=int(row["frame_id"]),
                    pos_xyz=pos,
                    R_body=R,
                )
            except KeyError as e:
                raise KeyError(
                    f"CSV {path!r} is missing column {e!s}. "
                    f"Expected columns: timestamp_sec, frame_id, "
                    f"pos_x, pos_y, pos_z, and one of: "
                    f"[qx,qy,qz,qw] | [roll,pitch,yaw] | [yaw]."
                ) from e
            rows.append(pose)
    return rows
