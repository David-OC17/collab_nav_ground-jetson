"""
arena_marker_localizer.optitrack
─────────────────────────────────────────────────────────────────────────────
Read OptiTrack drone pose log. The CSV is expected to be one row per
video frame in order:

    timestamp_sec,frame_id,pos_x,pos_y,pos_z,yaw

The yaw axis is configurable (Z-up or Y-up) so the same code handles
either convention. The +x, +y direction of the OptiTrack frame is also
configurable downstream via the transforms module — this reader stays
agnostic and just gives back the raw values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import csv
import numpy as np


@dataclass
class DronePose:
    timestamp_sec: float
    frame_id:      int
    pos_xyz:       np.ndarray   # (3,) float64
    yaw_rad:       float


def load_optitrack_csv(path: str) -> List[DronePose]:
    """Read the CSV, return one DronePose per row in file order."""
    rows: List[DronePose] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
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
                    yaw_rad=float(row["yaw"]),
                )
            except KeyError as e:
                raise KeyError(
                    f"CSV {path!r} is missing column {e!s}. "
                    f"Expected: timestamp_sec, frame_id, pos_x, pos_y, "
                    f"pos_z, yaw."
                ) from e
            rows.append(pose)
    return rows
