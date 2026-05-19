"""Conversions between ``nav_msgs/OccupancyGrid`` and numpy arrays, plus the
metric-coordinate helpers and the Stage 7 reprojection routine.
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import apply_se2, quaternion_from_yaw, yaw_from_quaternion


@dataclass
class GridInfo:
    """Geometry of an OccupancyGrid, decoupled from the message type."""

    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float

    @classmethod
    def from_msg(cls, info):
        o = info.origin
        return cls(
            resolution=float(info.resolution),
            width=int(info.width),
            height=int(info.height),
            origin_x=float(o.position.x),
            origin_y=float(o.position.y),
            origin_yaw=yaw_from_quaternion(
                o.orientation.x, o.orientation.y,
                o.orientation.z, o.orientation.w),
        )


def occupancygrid_to_array(msg):
    """Return ``(HxW int8 array, GridInfo)``. Cell values: -1, or 0..100."""
    info = GridInfo.from_msg(msg.info)
    arr = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
    return arr, info


def cell_centers_to_metric(rows, cols, info):
    """Vectorised (row, col) -> (N, 2) metric points in the grid's own frame.

    Uses cell *centres* and honours a non-zero ``origin_yaw``.
    """
    local = np.stack([(np.asarray(cols) + 0.5) * info.resolution,
                      (np.asarray(rows) + 0.5) * info.resolution], axis=-1)
    c, s = math.cos(info.origin_yaw), math.sin(info.origin_yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([info.origin_x, info.origin_y])


def array_to_occupancygrid(arr, info, frame_id, stamp):
    """Build a ``nav_msgs/OccupancyGrid`` message from an array + GridInfo."""
    from geometry_msgs.msg import Quaternion
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.resolution = float(info.resolution)
    msg.info.width = int(info.width)
    msg.info.height = int(info.height)
    msg.info.origin.position.x = float(info.origin_x)
    msg.info.origin.position.y = float(info.origin_y)
    qx, qy, qz, qw = quaternion_from_yaw(info.origin_yaw)
    msg.info.origin.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    msg.data = arr.astype(np.int8).flatten().tolist()
    return msg


def reproject_slam_grid(slam_arr, slam_info, t_world_slam, drone_info,
                        out_resolution):
    """Stage 7 reprojection: SLAM grid -> world-frame OccupancyGrid array.

    Each *known* SLAM cell is transformed into the world frame by
    ``t_world_slam`` and rasterised onto a grid that shares the drone map's
    origin and metric extent. To avoid holes when upsampling (the SLAM map is
    typically coarser, e.g. 0.05 -> 0.02 m/cell) each SLAM cell is splatted as
    a small square. Unknown cells (-1) are preserved: any output cell that
    receives no SLAM contribution stays -1.

    Returns ``(int8 array, GridInfo)`` in the ``world`` frame.
    """
    drone_w_m = drone_info.width * drone_info.resolution
    drone_h_m = drone_info.height * drone_info.resolution
    out_w = max(1, int(round(drone_w_m / out_resolution)))
    out_h = max(1, int(round(drone_h_m / out_resolution)))
    out_info = GridInfo(out_resolution, out_w, out_h,
                        drone_info.origin_x, drone_info.origin_y, 0.0)

    # int16 accumulator so np.maximum.at behaves; -1 is the "unknown" sentinel.
    out = np.full((out_h, out_w), -1, dtype=np.int16)

    rows, cols = np.where(slam_arr != -1)
    if rows.size == 0:
        return out.astype(np.int8), out_info

    vals = slam_arr[rows, cols].astype(np.int16)
    pts = cell_centers_to_metric(rows, cols, slam_info)
    world = apply_se2(t_world_slam, pts)
    base_c = np.floor((world[:, 0] - out_info.origin_x) / out_resolution).astype(int)
    base_r = np.floor((world[:, 1] - out_info.origin_y) / out_resolution).astype(int)

    splat = max(1, int(math.ceil(slam_info.resolution / out_resolution)))
    half = splat // 2
    for dy in range(splat):
        for dx in range(splat):
            rr = base_r + dy - half
            cc = base_c + dx - half
            ok = (rr >= 0) & (rr < out_h) & (cc >= 0) & (cc < out_w)
            np.maximum.at(out, (rr[ok], cc[ok]), vals[ok])

    return out.astype(np.int8), out_info
