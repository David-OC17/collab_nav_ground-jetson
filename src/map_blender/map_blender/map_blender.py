#!/usr/bin/env python3
"""
Map Merge Node
==============
Merges a fixed global map (loaded once at startup) with a live GMapping
probabilistic occupancy grid. Publishes a blended merged map and triggers
a replan when a new obstacle appears within a configurable proximity of
the robot.

Architecture:
  /map_global  (latched, published once at startup)  ─┐
                                                       ├─► MapMergeNode ─► /map_merged ──► A* planner
  /map_gmapping (live, updated by GMapping)          ─┘                ─► /replan      ──► triggers replan

Subscribes to:
  - /map_global    (nav_msgs/OccupancyGrid) — fixed global map
  - /map_gmapping  (nav_msgs/OccupancyGrid) — live GMapping map

Uses:
  - TF (map → base_link) — robot position for proximity check

Publishes:
  - /map_merged              (nav_msgs/OccupancyGrid) — blended map
  - /map_merge/new_obstacles (visualization_msgs/MarkerArray) — RViz markers
  - /map_merge/replan        (std_msgs/Bool) — pulse True when replan needed

Parameters:
  - global_map_topic    (str,   default '/map_global')
  - gmapping_map_topic  (str,   default '/map_gmapping')
  - merged_map_topic    (str,   default '/map_merged')
  - map_frame           (str,   default 'map')
  - robot_base_frame    (str,   default 'base_link')
  - global_weight       (float, default 0.9)  weight for global map cells
  - gmapping_weight     (float, default 0.1)  weight for GMapping cells
  - new_obstacle_thresh (int,   default 65)   occupancy value to call a cell an obstacle
  - replan_distance     (float, default 1.0)  metres — replan if new obstacle within this
  - publish_rate        (float, default 2.0)  Hz — merged map publish rate
  - change_threshold    (int,   default 10)   min occupancy delta to flag a cell as changed
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class MapMergeNode(Node):

    def __init__(self):
        super().__init__('map_merge_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('global_map_topic',    '/map_global')
        self.declare_parameter('gmapping_map_topic',  '/map_gmapping')
        self.declare_parameter('merged_map_topic',    '/map_merged')
        self.declare_parameter('map_frame',           'map')
        self.declare_parameter('robot_base_frame',    'base_link')
        self.declare_parameter('global_weight',       0.9)
        self.declare_parameter('gmapping_weight',     0.1)
        self.declare_parameter('new_obstacle_thresh', 65)
        self.declare_parameter('replan_distance',     1.0)
        self.declare_parameter('publish_rate',        2.0)
        self.declare_parameter('change_threshold',    10)

        self.global_map_topic    = self.get_parameter('global_map_topic').value
        self.gmapping_map_topic  = self.get_parameter('gmapping_map_topic').value
        self.merged_map_topic    = self.get_parameter('merged_map_topic').value
        self.map_frame           = self.get_parameter('map_frame').value
        self.robot_base_frame    = self.get_parameter('robot_base_frame').value
        self.global_weight       = self.get_parameter('global_weight').value
        self.gmapping_weight     = self.get_parameter('gmapping_weight').value
        self.new_obstacle_thresh = self.get_parameter('new_obstacle_thresh').value
        self.replan_distance     = self.get_parameter('replan_distance').value
        self.publish_rate        = self.get_parameter('publish_rate').value
        self.change_threshold    = self.get_parameter('change_threshold').value

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        # Global map (fixed at startup)
        self.global_data      = None   # np.ndarray (H, W) int8
        self.global_res       = None
        self.global_origin_x  = None
        self.global_origin_y  = None
        self.global_width     = None
        self.global_height    = None
        self.global_received  = False

        # GMapping map (live)
        self.gmap_data        = None
        self.gmap_res         = None
        self.gmap_origin_x    = None
        self.gmap_origin_y    = None
        self.gmap_width       = None
        self.gmap_height      = None
        self.gmap_received    = False

        # Previous merged map for change detection
        self.prev_merged      = None

        # Robot pose
        self.robot_x          = 0.0
        self.robot_y          = 0.0

        # Replan cooldown — avoid flooding the planner
        self.last_replan_time = 0.0
        self.replan_cooldown  = 3.0   # seconds

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.global_sub = self.create_subscription(
            OccupancyGrid,
            self.global_map_topic,
            self._global_map_callback,
            latched_qos
        )

        self.gmap_sub = self.create_subscription(
            OccupancyGrid,
            self.gmapping_map_topic,
            self._gmapping_callback,
            latched_qos
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.merged_pub = self.create_publisher(
            OccupancyGrid, self.merged_map_topic, latched_qos)

        self.replan_pub = self.create_publisher(
            Bool, '/map_merge/replan', reliable_qos)

        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, '/map_merge/new_obstacles', reliable_qos)

        # ------------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------------
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self._update)

        self.get_logger().info(
            f'MapMergeNode ready | '
            f'global={self.global_map_topic} | '
            f'gmapping={self.gmapping_map_topic} | '
            f'blend={self.global_weight:.1f}/{self.gmapping_weight:.1f} | '
            f'replan_dist={self.replan_distance}m'
        )

    # ==========================================================================
    # Map callbacks
    # ==========================================================================

    def _global_map_callback(self, msg: OccupancyGrid):
        """Store the global map — received once at startup."""
        if self.global_received:
            return   # ignore subsequent publishes — global map is fixed

        self.global_res      = msg.info.resolution
        self.global_origin_x = msg.info.origin.position.x
        self.global_origin_y = msg.info.origin.position.y
        self.global_width    = msg.info.width
        self.global_height   = msg.info.height
        self.global_data     = np.array(msg.data, dtype=np.int8).reshape(
            (self.global_height, self.global_width))
        self.global_received = True

        self.get_logger().info(
            f'Global map received: {self.global_width}x{self.global_height} cells '
            f'@ {self.global_res}m/cell — locked.')

    def _gmapping_callback(self, msg: OccupancyGrid):
        """Update the live GMapping map."""
        self.gmap_res      = msg.info.resolution
        self.gmap_origin_x = msg.info.origin.position.x
        self.gmap_origin_y = msg.info.origin.position.y
        self.gmap_width    = msg.info.width
        self.gmap_height   = msg.info.height
        self.gmap_data     = np.array(msg.data, dtype=np.int8).reshape(
            (self.gmap_height, self.gmap_width))
        self.gmap_received = True

        self.get_logger().debug(
            f'GMapping map updated: {self.gmap_width}x{self.gmap_height} cells',
            throttle_duration_sec=5.0)

    # ==========================================================================
    # Main update loop
    # ==========================================================================

    def _update(self):
        if not self.global_received:
            self.get_logger().warn(
                f'Waiting for global map on {self.global_map_topic}...',
                throttle_duration_sec=3.0)
            return

        # Get robot pose
        self._update_robot_pose()

        # Merge maps
        merged = self._merge()

        # Detect new obstacles vs previous merged map
        if self.prev_merged is not None:
            new_obstacle_cells = self._detect_new_obstacles(merged)
            if new_obstacle_cells:
                self._publish_obstacle_markers(new_obstacle_cells)
                self._check_replan_trigger(new_obstacle_cells)

        self.prev_merged = merged.copy()

        # Publish merged map on global frame
        self._publish_merged(merged)

    # ==========================================================================
    # Map merging
    # ==========================================================================

    def _merge(self) -> np.ndarray:
        """
        Blend global and GMapping maps into one.

        Strategy:
          - Start with the global map as the base
          - For each cell in the GMapping map that has been observed (val >= 0),
            compute a weighted blend:
              merged = global_weight * global_val + gmapping_weight * gmap_val
          - Unknown GMapping cells (val == -1) leave the global map unchanged
          - GMapping cells that extend beyond the global map boundary are ignored
            (global map size is fixed)
        """
        # Work in float32 for blending, clip to int8 at the end
        merged = self.global_data.astype(np.float32).copy()

        if not self.gmap_received:
            return merged.astype(np.int8)

        # Iterate over GMapping cells and project onto global map frame
        for gcj in range(self.gmap_height):
            for gci in range(self.gmap_width):
                gval = int(self.gmap_data[gcj, gci])
                if gval < 0:
                    continue   # unobserved — skip

                # World coordinates of this GMapping cell
                wx = self.gmap_origin_x + (gci + 0.5) * self.gmap_res
                wy = self.gmap_origin_y + (gcj + 0.5) * self.gmap_res

                # Corresponding cell in global map
                ci = int((wx - self.global_origin_x) / self.global_res)
                cj = int((wy - self.global_origin_y) / self.global_res)

                if not (0 <= ci < self.global_width and
                        0 <= cj < self.global_height):
                    continue   # outside global map boundary

                glo_val = float(self.global_data[cj, ci])
                if glo_val < 0:
                    glo_val = 0.0   # treat unknown global as free

                blended = (self.global_weight   * glo_val +
                           self.gmapping_weight * float(gval))
                merged[cj, ci] = np.clip(blended, 0, 100)

        return merged.astype(np.int8)

    # ==========================================================================
    # New obstacle detection
    # ==========================================================================

    def _detect_new_obstacles(self, merged: np.ndarray) -> list:
        """
        Compare merged map to previous merged map.
        Returns list of (world_x, world_y) for cells that:
          1. Are newly occupied (above new_obstacle_thresh)
          2. Were not occupied before (delta > change_threshold)
        """
        # Cells newly above threshold
        curr_occ = merged.astype(np.int16)
        prev_occ = self.prev_merged.astype(np.int16)

        delta = curr_occ - prev_occ

        # New obstacle: now above threshold AND increased significantly
        new_mask = (
            (curr_occ >= self.new_obstacle_thresh) &
            (delta     >= self.change_threshold)
        )

        new_cells_idx = np.argwhere(new_mask)   # (cj, ci) pairs
        if len(new_cells_idx) == 0:
            return []

        # Convert to world coordinates
        new_obstacle_world = []
        for (cj, ci) in new_cells_idx:
            wx = self.global_origin_x + (ci + 0.5) * self.global_res
            wy = self.global_origin_y + (cj + 0.5) * self.global_res
            new_obstacle_world.append((float(wx), float(wy)))

        self.get_logger().info(
            f'Detected {len(new_obstacle_world)} new obstacle cells')
        return new_obstacle_world

    # ==========================================================================
    # Replan trigger
    # ==========================================================================

    def _check_replan_trigger(self, new_obstacles: list):
        """
        Trigger a replan if any new obstacle is within replan_distance
        of the robot. Cooldown prevents flooding the planner.
        """
        now = self.get_clock().now().nanoseconds / 1e9

        if (now - self.last_replan_time) < self.replan_cooldown:
            return   # still in cooldown

        for wx, wy in new_obstacles:
            dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
            if dist <= self.replan_distance:
                self.get_logger().warn(
                    f'New obstacle at ({wx:.2f},{wy:.2f}) is {dist:.2f}m '
                    f'from robot — triggering replan!')
                msg = Bool()
                msg.data = True
                self.replan_pub.publish(msg)
                self.last_replan_time = now
                return   # one replan per cooldown window

    # ==========================================================================
    # Robot pose
    # ==========================================================================

    def _update_robot_pose(self):
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            self.robot_x = tf_stamped.transform.translation.x
            self.robot_y = tf_stamped.transform.translation.y
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass   # keep last known pose

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _publish_merged(self, merged: np.ndarray):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.global_res
        msg.info.width      = self.global_width
        msg.info.height     = self.global_height
        msg.info.origin.position.x  = self.global_origin_x
        msg.info.origin.position.y  = self.global_origin_y
        msg.info.origin.position.z  = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = merged.flatten().tolist()
        self.merged_pub.publish(msg)

    def _publish_obstacle_markers(self, new_obstacles: list):
        ma = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.stamp    = self.get_clock().now().to_msg()
        clear.header.frame_id = self.map_frame
        clear.ns     = 'new_obstacles'
        clear.id     = 0
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # New obstacle spheres
        for idx, (wx, wy) in enumerate(new_obstacles[:50]):   # cap at 50 markers
            m = Marker()
            m.header.stamp    = self.get_clock().now().to_msg()
            m.header.frame_id = self.map_frame
            m.ns     = 'new_obstacles'
            m.id     = idx + 1
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x    = wx
            m.pose.position.y    = wy
            m.pose.position.z    = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.12

            # Colour by distance to robot
            dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
            t    = min(dist / self.replan_distance, 1.0)
            m.color.r = 1.0 - t    # red when close, fades with distance
            m.color.g = 0.2
            m.color.b = t
            m.color.a = 0.9
            ma.markers.append(m)

        self.obstacle_marker_pub.publish(ma)


# ==============================================================================
# Entry point
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = MapMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()