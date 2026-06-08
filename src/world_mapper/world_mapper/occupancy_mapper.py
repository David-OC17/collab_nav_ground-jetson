#!/usr/bin/env python3
"""
occupancy_mapper.py

Builds an occupancy grid in the `world` frame from LaserScan data, using
slam_toolbox (via the TF tree) purely as a drift-corrected pose source.

Frame chain assumed (none of these are published by THIS node):
    world -> slam_map -> odom -> base_footprint -> laser_frame
      ^         ^          ^
      |         |          slam_toolbox
      |         alignment_node (you)
      world is fixed/global

This node only LISTENS to TF. It looks up `world -> laser_frame` at each
scan timestamp, raycasts every beam into a log-odds grid, and periodically
publishes a nav_msgs/OccupancyGrid with frame_id="world".
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy

import tf2_ros
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


def yaw_from_quaternion(x, y, z, w):
    """Extract the planar yaw angle from a quaternion (2D mapper only cares about yaw)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def bresenham(x0, y0, x1, y1):
    """Integer Bresenham line. Yields every cell from (x0,y0) to (x1,y1) inclusive."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


class OccupancyMapper(Node):
    def __init__(self):
        super().__init__("occupancy_mapper")

        # ---- Parameters -------------------------------------------------
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("laser_frame", "lidar")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/amr/world_map")

        self.declare_parameter("resolution", 0.05)       # meters / cell
        self.declare_parameter("width_m", 3.9)          # map width in meters
        self.declare_parameter("height_m", 3.9)         # map height in meters
        # Pose of cell (0,0) (bottom-left corner) in the world frame.
        # Default centers the grid on the world origin. If your arena lives
        # entirely in the +x/+y quadrant, set both to 0.0 instead.
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)

        # Log-odds update model
        self.declare_parameter("l_occ", 0.85)            # add when a cell is hit
        self.declare_parameter("l_free", -0.40)          # add when a cell is passed through
        self.declare_parameter("l_min", -5.0)            # clamp (free saturation)
        self.declare_parameter("l_max", 5.0)             # clamp (occupied saturation)

        self.declare_parameter("publish_rate", 1.0)      # Hz
        self.declare_parameter("tf_timeout", 0.10)       # seconds to wait for a transform

        gp = self.get_parameter
        self.world_frame = gp("world_frame").value
        self.laser_frame = gp("laser_frame").value
        self.scan_topic = gp("scan_topic").value
        self.map_topic = gp("map_topic").value

        self.res = float(gp("resolution").value)
        self.origin_x = float(gp("origin_x").value)
        self.origin_y = float(gp("origin_y").value)
        self.width = int(round(float(gp("width_m").value) / self.res))
        self.height = int(round(float(gp("height_m").value) / self.res))

        self.l_occ = float(gp("l_occ").value)
        self.l_free = float(gp("l_free").value)
        self.l_min = float(gp("l_min").value)
        self.l_max = float(gp("l_max").value)
        self.tf_timeout = rclpy.duration.Duration(seconds=float(gp("tf_timeout").value))

        # ---- State ------------------------------------------------------
        # Log-odds grid, row-major: index = row * width + col.
        # Prior is 0.5 -> log-odds 0.0, which we treat as "unknown" on publish.
        self.logodds = np.zeros(self.height * self.width, dtype=np.float32)
        self.lock = threading.Lock()

        # ---- TF ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- I/O --------------------------------------------------------
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data
        )
        # Transient-local so late-joining RViz/Nav2 still receive the last map.
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, map_qos)

        period = 1.0 / float(gp("publish_rate").value)
        self.timer = self.create_timer(period, self.publish_map)

        self.get_logger().info(
            f"OccupancyMapper up. {self.width}x{self.height} @ {self.res} m/cell, "
            f"origin=({self.origin_x:.2f},{self.origin_y:.2f}) in '{self.world_frame}'. "
            f"Raycasting '{self.laser_frame}' -> grid."
        )

    # ---------------------------------------------------------------------
    def world_to_cell(self, x, y):
        """World (m) -> grid (col, row). Returns None if out of bounds."""
        col = int(math.floor((x - self.origin_x) / self.res))
        row = int(math.floor((y - self.origin_y) / self.res))
        if 0 <= col < self.width and 0 <= row < self.height:
            return col, row
        return None

    def clamp_to_grid(self, col, row):
        """Clamp a (possibly out-of-bounds) cell to the grid edge, so a long
        ray still marks free space up to the boundary it exits through."""
        return (min(max(col, 0), self.width - 1),
                min(max(row, 0), self.height - 1))

    # ---------------------------------------------------------------------
    def scan_cb(self, scan: LaserScan):
        # Look up the laser pose in the world frame AT THE SCAN'S TIMESTAMP.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                scan.header.frame_id or self.laser_frame,
                rclpy.time.Time(),
                timeout=self.tf_timeout,
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(
                f"TF {self.world_frame} <- {scan.header.frame_id} unavailable, "
                f"dropping scan: {e}", throttle_duration_sec=2.0
            )
            return

        ox = tf.transform.translation.x
        oy = tf.transform.translation.y
        q = tf.transform.rotation
        oyaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        origin_cell = self.world_to_cell(ox, oy)
        if origin_cell is None:
            self.get_logger().warn(
                "Sensor is outside the map bounds; growing the map or moving "
                "origin_x/origin_y is needed.", throttle_duration_sec=5.0
            )
            return
        ocol, orow = origin_cell

        # Accumulate updates locally, apply once under the lock.
        hits = []     # cells to mark occupied
        misses = []   # cells to mark free

        rmin = scan.range_min
        rmax = scan.range_max
        angle = scan.angle_min
        for r in scan.ranges:
            a = angle
            angle += scan.angle_increment

            # Skip invalid readings.
            if r is None or math.isnan(r) or math.isinf(r):
                continue
            if r < rmin:
                continue

            is_max = r >= rmax
            rr = min(r, rmax)

            # Ray endpoint in world coordinates.
            world_a = oyaw + a
            ex = ox + rr * math.cos(world_a)
            ey = oy + rr * math.sin(world_a)

            end_cell = self.world_to_cell(ex, ey)
            if end_cell is None:
                # Endpoint left the map: clamp so we still free-mark up to the edge,
                # but DON'T record an occupied hit (we never saw the true endpoint).
                ecol, erow = self.clamp_to_grid(
                    int(math.floor((ex - self.origin_x) / self.res)),
                    int(math.floor((ey - self.origin_y) / self.res)),
                )
                is_max = True
            else:
                ecol, erow = end_cell

            # Walk the ray. All cells before the endpoint are free.
            cells = list(bresenham(ocol, orow, ecol, erow))
            for c, rrow in cells[:-1]:
                misses.append(rrow * self.width + c)
            # Endpoint: occupied only if it was a real return (not max-range).
            last_c, last_r = cells[-1]
            if not is_max:
                hits.append(last_r * self.width + last_c)
            else:
                misses.append(last_r * self.width + last_c)

        if not hits and not misses:
            return

        with self.lock:
            if misses:
                idx = np.fromiter(misses, dtype=np.intp)
                np.add.at(self.logodds, idx, self.l_free)
            if hits:
                idx = np.fromiter(hits, dtype=np.intp)
                np.add.at(self.logodds, idx, self.l_occ)
            np.clip(self.logodds, self.l_min, self.l_max, out=self.logodds)

    # ---------------------------------------------------------------------
    def publish_map(self):
        with self.lock:
            l = self.logodds.copy()

        # log-odds -> probability -> 0..100, with l==0 meaning "unknown" (-1).
        prob = 1.0 / (1.0 + np.exp(-l))          # p = 1/(1+e^-l)
        data = np.rint(prob * 100.0).astype(np.int8)
        data[l == 0.0] = -1                      # never observed

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.res
        msg.info.width = self.width
        msg.info.height = self.height
        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.orientation.w = 1.0
        msg.info.origin = origin
        msg.data = data.tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()