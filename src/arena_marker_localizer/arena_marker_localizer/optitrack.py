"""
arena_marker_localizer.optitrack
─────────────────────────────────────────────────────────────────────────────
Read OptiTrack drone pose log.

Full-pose CSV format (preferred):
    timestamp_sec,frame_id,pos_x,pos_y,pos_z,roll,pitch,yaw

Legacy format (yaw-only, backward-compatible):
    timestamp_sec,frame_id,pos_x,pos_y,pos_z,yaw

All angles are in radians, Tait-Bryan ZYX convention (same as OptiTrack's
default Euler output when configured to ZYX order).  When the CSV lacks
'roll' and 'pitch' columns both default to 0.0 and a one-time warning is
emitted.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DronePose:
    timestamp_sec: float
    frame_id:      int
    pos_xyz:       np.ndarray   # (3,) float64
    roll_rad:      float        # 0.0 when CSV lacks 'roll' column
    pitch_rad:     float        # 0.0 when CSV lacks 'pitch' column
    yaw_rad:       float


def load_optitrack_csv(path: str) -> List[DronePose]:
    """Read the CSV; return one DronePose per row in file order.

    Accepts both the full-pose format (with 'roll' and 'pitch' columns) and
    the legacy yaw-only format.  Missing roll/pitch default to 0.0 with a
    one-time deprecation warning.
    """
    rows: List[DronePose] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        has_roll  = "roll"  in fieldnames
        has_pitch = "pitch" in fieldnames

        if not (has_roll and has_pitch):
            warnings.warn(
                f"CSV {path!r} is missing 'roll' and/or 'pitch' columns. "
                "Defaulting both to 0.0 (legacy yaw-only mode). "
                "Export full 6-DoF pose from OptiTrack for better accuracy.",
                UserWarning,
                stacklevel=2,
            )

        for row in reader:
            try:
                pose = DronePose(
                    timestamp_sec=float(row["timestamp_sec"]),
                    frame_id=int(row["frame_id"]),
                    pos_xyz=np.array(
                        [float(row["pos_x"]),
                         float(row["pos_y"]),
                         float(row["pos_z"])],
                        dtype=np.float64,
                    ),
                    roll_rad=float(row["roll"])  if has_roll  else 0.0,
                    pitch_rad=float(row["pitch"]) if has_pitch else 0.0,
                    yaw_rad=float(row["yaw"]),
                )
            except KeyError as e:
                raise KeyError(
                    f"CSV {path!r} is missing column {e!s}. "
                    f"Expected columns: timestamp_sec, frame_id, "
                    f"pos_x, pos_y, pos_z, [roll, pitch,] yaw."
                ) from e
            rows.append(pose)
    return rows
