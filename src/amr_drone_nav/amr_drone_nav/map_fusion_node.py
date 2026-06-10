#!/usr/bin/env python3
"""map_fusion_node.py — Fuses drone map and SLAM map into a single OccupancyGrid
published in the world frame.

Pipeline:
  1. Receives /drone/map  (OccupancyGrid in world frame)
  2. Receives /map        (OccupancyGrid in slam_map frame)
  3. Reads TF world→slam_map to reproject SLAM cells into world frame
  4. Fuses both grids into /map_fused (OccupancyGrid in world frame)

Fusion rule (per output cell in world frame):
  SLAM known  (0 or 100) → SLAM wins
  SLAM unknown (-1)       → drone value fills (may also be -1)

Reprojection of SLAM map:
  For each output cell center (x_w, y_w) in world:
    (x_s, y_s) = R^T * ((x_w, y_w) - (tx, ty))   ← inv(T_world_slam)
    find nearest cell in SLAM grid at (x_s, y_s)

Topics
------
  Sub: /drone/map   nav_msgs/OccupancyGrid  latched
  Sub: /map         nav_msgs/OccupancyGrid  latched
  Pub: /map_fused   nav_msgs/OccupancyGrid  latched  (world frame)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy,
    QoSProfile, QoSReliabilityPolicy,
)
from nav_msgs.msg import OccupancyGrid
import tf2_ros

_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
)

PARAM_SPEC = [
    ("drone_map_topic",          "/drone/map"),
    ("slam_map_topic",           "/map"),
    ("fused_map_topic",          "/map_fused"),
    ("world_frame",              "world"),
    ("slam_frame",               "slam_map"),
    ("output_resolution",        0.05),  # m/cell
    ("output_origin_x",          0.0),   # world frame origin of output grid
    ("output_origin_y",          0.0),
    ("output_width",             4.0),   # meters
    ("output_height",            4.0),   # meters
    ("drone_occupied_threshold", 65),
    ("drone_free_threshold",     20),
    ("tf_timeout",               1.0),
]


def quat_to_yaw(r):
    return math.atan2(2*(r.w*r.z + r.x*r.y), 1 - 2*(r.y*r.y + r.z*r.z))


class MapFusionNode(Node):

    def __init__(self):
        super().__init__("map_fusion_node")

        self.p = {}
        for name, default in PARAM_SPEC:
            self.declare_parameter(name, default)
            self.p[name] = self.get_parameter(name).value

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._drone_arr  = None
        self._drone_info = None
        self._slam_arr   = None
        self._slam_info  = None

        self._pub = self.create_publisher(
            OccupancyGrid, self.p["fused_map_topic"], _QOS
        )
        self.create_subscription(
            OccupancyGrid, self.p["drone_map_topic"], self._drone_cb, _QOS
        )
        self.create_subscription(
            OccupancyGrid, self.p["slam_map_topic"],  self._slam_cb,  _QOS
        )

        self.get_logger().info("map_fusion_node started")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _tf_world_to_slam(self):
        """Returns (tx, ty, yaw) of T(world→slam_map) or None."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self.p["world_frame"], self.p["slam_frame"],
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.p["tf_timeout"]))
            t = tf.transform.translation
            return t.x, t.y, quat_to_yaw(tf.transform.rotation)
        except Exception as e:
            self.get_logger().warn(f"TF not available: {e}")
            return None

    def _discretise(self, arr):
        out = np.full_like(arr, -1, dtype=np.int8)
        out[arr >= self.p["drone_occupied_threshold"]] = 100
        out[arr <= self.p["drone_free_threshold"]]     = 0
        return out

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _drone_cb(self, msg: OccupancyGrid):
        self._drone_arr  = self._discretise(
            np.array(msg.data, dtype=np.int8).reshape(
                msg.info.height, msg.info.width))
        self._drone_info = msg.info
        self.get_logger().info(
            f"[drone] {msg.info.width}×{msg.info.height} received")
        self._try_fuse()

    def _slam_cb(self, msg: OccupancyGrid):
        self._slam_arr  = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._slam_info = msg.info
        self.get_logger().info(
            f"[slam] {msg.info.width}×{msg.info.height} received")
        self._try_fuse()

    # ── fusion ────────────────────────────────────────────────────────────────

    def _try_fuse(self):
        if self._drone_arr is None or self._slam_arr is None:
            return
        tf = self._tf_world_to_slam()
        if tf is None:
            return
        self._fuse(tf)

    def _fuse(self, tf):
        res    = self.p["output_resolution"]
        ox     = self.p["output_origin_x"]
        oy     = self.p["output_origin_y"]
        cols_n = int(round(self.p["output_width"]  / res))
        rows_n = int(round(self.p["output_height"] / res))

        # T(world→slam_map): translation + rotation
        tx_ws, ty_ws, yaw_ws = tf
        c =  math.cos(yaw_ws)
        s =  math.sin(yaw_ws)

        # Output cell centers in world frame
        # x_w[col] = ox + (col + 0.5) * res
        # y_w[row] = oy + (row + 0.5) * res
        x_w = ox + (np.arange(cols_n) + 0.5) * res   # (cols_n,)
        y_w = oy + (np.arange(rows_n) + 0.5) * res   # (rows_n,)

        # inv(T_world_slam): p_slam = R^T * (p_world - t)
        # x_s =  c*(x_w - tx_ws) + s*(y_w - ty_ws)
        # y_s = -s*(x_w - tx_ws) + c*(y_w - ty_ws)
        dx = x_w - tx_ws   # (cols_n,)
        dy = y_w - ty_ws   # (rows_n,)

        # Meshgrid for vectorized computation
        DX, DY = np.meshgrid(dx, dy)   # both (rows_n, cols_n)
        X_s =  c * DX + s * DY        # SLAM x for every output cell
        Y_s = -s * DX + c * DY        # SLAM y for every output cell

        # ── Layer 1: drone (already in world frame) ───────────────────────────
        di = self._drone_info
        dc = np.round(
            (x_w - di.origin.position.x) / di.resolution - 0.5
        ).astype(int)   # (cols_n,)
        dr = np.round(
            (y_w - di.origin.position.y) / di.resolution - 0.5
        ).astype(int)   # (rows_n,)

        DC, DR = np.meshgrid(dc, dr)   # (rows_n, cols_n)

        valid_d = ((DC >= 0) & (DC < di.width) &
                   (DR >= 0) & (DR < di.height))

        out = np.full((rows_n, cols_n), -1, dtype=np.int8)
        out[valid_d] = self._drone_arr[DR[valid_d], DC[valid_d]]

        # ── Layer 2: SLAM (reprojected via TF) ───────────────────────────────
        si = self._slam_info
        SC = np.round(
            (X_s - si.origin.position.x) / si.resolution - 0.5
        ).astype(int)   # (rows_n, cols_n)
        SR = np.round(
            (Y_s - si.origin.position.y) / si.resolution - 0.5
        ).astype(int)   # (rows_n, cols_n)

        valid_s = ((SC >= 0) & (SC < si.width) &
                   (SR >= 0) & (SR < si.height))

        slam_vals = np.full((rows_n, cols_n), -1, dtype=np.int8)
        slam_vals[valid_s] = self._slam_arr[SR[valid_s], SC[valid_s]]

        # SLAM wins where it has known data
        known = slam_vals != -1
        out[known] = slam_vals[known]

        # ── Publish ───────────────────────────────────────────────────────────
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.p["world_frame"]
        msg.info.resolution = res
        msg.info.width      = cols_n
        msg.info.height     = rows_n
        msg.info.origin.position.x    = ox
        msg.info.origin.position.y    = oy
        msg.info.origin.orientation.w = 1.0
        msg.data = out.flatten(order='C').tolist()
        self._pub.publish(msg)

        n_known = int((out != -1).sum())
        self.get_logger().info(
            f"[fused] {cols_n}×{rows_n} | "
            f"TF yaw={math.degrees(yaw_ws):.1f}° | "
            f"coverage {100*n_known/(cols_n*rows_n):.0f}%"
        )


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapFusionNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
