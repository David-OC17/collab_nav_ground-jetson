#!/usr/bin/env python3
"""
Frontier Explorer Node
========================================
Detects frontiers on the fused occupancy grid, clusters them, scores each
cluster by a weighted combination of travel distance and information gain
(cluster area), and publishes the best frontier centroid as a navigation goal.

A frontier cell is any FREE cell (value == 0) that has at least one
UNKNOWN neighbour (value == -1).  Clusters are built with a simple
union-find over 8-connected frontier cells.

Scoring (higher is better):
  score = w_dist * (1 / distance_m) + w_size * cluster_area_m2

  w_dist   — weight for proximity  (default 0.7): favours closer frontiers
  w_size   — weight for area       (default 0.3): favours larger unexplored regions
  cluster_area_m2 = len(cells) * resolution²

The centroid is validated and snapped to the nearest real frontier cell
if the raw average does not land on a free navigable cell.

Subscribes:
  - /drone/map      (nav_msgs/OccupancyGrid)                   — fused occupancy map
  - /follower/pose  (geometry_msgs/PoseWithCovarianceStamped)  — robot pose

Publishes:
  - /frontier/goal     (geometry_msgs/PoseWithCovarianceStamped) — best frontier centroid
  - /frontier/markers  (visualization_msgs/MarkerArray)          — RViz visualisation

Parameters:
  map_topic             '/drone/map'
  pose_topic            '/follower/pose'
  goal_topic            '/frontier/goal'
  marker_topic          '/frontier/markers'
  world_frame           'map'
  min_cluster_size      5       cells  — discard tiny frontier clusters
  max_frontier_dist     8.0     m      — ignore frontiers farther than this
  update_rate           1.0     Hz     — how often to recompute frontiers
  goal_reached_dist     0.40    m      — re-explore when robot is this close to goal
  w_dist                0.7            — scoring weight for 1/distance
  w_size                0.3            — scoring weight for cluster area (m²)
  active                True           — set False to silence (MissionController uses this)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool
import tf2_ros
import tf2_geometry_msgs

from ros2_security import SecureNodeMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FREE     =  0
UNKNOWN  = -1
OCCUPIED = 100
LETHAL_THRESHOLD = 50   # cells above this are treated as occupied


class FrontierExplorer(SecureNodeMixin, Node):

    def __init__(self):
        super().__init__('frontier_explorer')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('map_topic',        '/drone/map')
        self.declare_parameter('pose_topic',       '/follower/pose')
        self.declare_parameter('goal_topic',       '/frontier/goal')
        self.declare_parameter('marker_topic',     '/frontier/markers')
        self.declare_parameter('world_frame',      'map')
        self.declare_parameter('min_cluster_size',  5)
        self.declare_parameter('max_frontier_dist', 10.0)
        self.declare_parameter('update_rate',       1.0)
        self.declare_parameter('w_dist',            0.7)
        self.declare_parameter('w_size',            0.3)
        self.declare_parameter('w_heading',         0.4)
        self.declare_parameter('active',            True)
        self.declare_parameter('odom_topic', '')   # Switch between simulation and real robot
        self.declare_parameter('safe_goal_radius', 0.30)   # m — must be >= astar inflation_radius
        self.declare_parameter('min_goal_dist',    0.20) 
        self.declare_parameter('goal_reached_dist', 0.12)   # just above spline tolerance of 0.1 m
        self.declare_parameter('require_camera_coverage', True)
        # Topic published by camera_fov_tracker; set empty string to disable
        self.declare_parameter('fov_map_topic', '/camera/fov_map')

        self.map_topic               = self.get_parameter('map_topic').value
        self.pose_topic              = self.get_parameter('pose_topic').value
        self.goal_topic              = self.get_parameter('goal_topic').value
        self.marker_topic            = self.get_parameter('marker_topic').value
        self.world_frame             = self.get_parameter('world_frame').value
        self.min_cluster_size        = int(self.get_parameter('min_cluster_size').value)
        self.max_frontier_dist       = float(self.get_parameter('max_frontier_dist').value)
        self.update_rate             = float(self.get_parameter('update_rate').value)
        self.w_dist                  = float(self.get_parameter('w_dist').value)
        self.w_size                  = float(self.get_parameter('w_size').value)
        self.w_heading               = float(self.get_parameter('w_heading').value)
        self.active                  = bool(self.get_parameter('active').value)
        self.odom_topic              = self.get_parameter('odom_topic').value
        self.safe_goal_radius        = float(self.get_parameter('safe_goal_radius').value)
        self.min_goal_dist           = float(self.get_parameter('min_goal_dist').value)
        self.goal_reached_dist       = float(self.get_parameter('goal_reached_dist').value)
        self.require_camera_coverage = bool(self.get_parameter('require_camera_coverage').value)
        self.fov_map_topic           = self.get_parameter('fov_map_topic').value

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.map_data       = None   # np.ndarray (height, width) int8 — SLAM map
        self.map_resolution = None
        self.map_origin_x   = None
        self.map_origin_y   = None
        self.map_width      = None
        self.map_height     = None
        self.map_received   = False

        # Drone map — used by _safe_goal to validate against A*'s actual map
        self.drone_map_data    = None   # np.ndarray (height, width) int8
        self.drone_map_received = False

        self.robot_x        = 0.0
        self.robot_y        = 0.0
        self.robot_yaw      = 0.0   # radians, updated from pose/odom callbacks
        self.pose_received  = False

        self.current_goal_x = None
        self.current_goal_y = None

        self.visited_goals         = []   # failure blacklist — A* said lethal
        self.goal_blacklist_radius = 0.30

        self.reached_goals            = []   # list of (wx, wy, timestamp)
        self.reached_blacklist_radius = 0.20  # m
        self.reached_blacklist_ttl    = 30.0  # s — expire after this long

        self._reset_state()

        # Prevent re-publishing the same goal every tick
        self._last_published_goal: tuple | None = None

        # Camera FOV mask — received from camera_fov_tracker
        # bool array (height, width); None until first message arrives
        self.fov_map_data:       np.ndarray | None = None
        self.fov_map_resolution: float | None      = None
        self.fov_map_origin_x:   float | None      = None
        self.fov_map_origin_y:   float | None      = None
        self.fov_map_width:      int   | None      = None
        self.fov_map_height:     int   | None      = None

        # TF buffer — used when odom_topic is set (real robot mode)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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
        self.map_sub = self.create_secure_subscription(
            self.map_topic, OccupancyGrid, self._map_callback, min_level=None, qos=latched_qos
        )

        # Drone map — for safe goal validation (same map A* uses)
        self.create_secure_subscription(
            '/drone/map', OccupancyGrid, self._drone_map_callback, min_level=None, qos=latched_qos
        )
        
        # Pose subscriber — two modes:
        #   odom_topic set   → real robot, subscribe to EKF Odometry + TF transform
        #   odom_topic empty → simulation, subscribe to /follower/pose directly
        if self.odom_topic:
            self.pose_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self._odom_callback,
                reliable_qos
            )
            self.get_logger().info(
                f'Pose source: {self.odom_topic} → transformed to {self.world_frame} via TF')
        else:
            self.pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                self.pose_topic,
                self._pose_callback,
                reliable_qos
            )
            self.get_logger().info(
                f'Pose source: {self.pose_topic} (direct)')
            
        # Planner failed to create a valid trajectory?
        self.create_secure_subscription(
            '/astar/goal_failed', PoseWithCovarianceStamped, self._goal_failed_callback, min_level=None, qos=reliable_qos
        )

        # MissionController can silence/activate this node at runtime
        self.active_sub = self.create_secure_subscription(
            '/frontier_explorer/active', Bool, self._active_callback, min_level=None, qos=reliable_qos
        )

        # Camera FOV mask — published by camera_fov_tracker
        if self.fov_map_topic:
            self.create_subscription(
                OccupancyGrid,
                self.fov_map_topic,
                self._fov_map_callback,
                latched_qos,
            )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.goal_pub = self.create_secure_publisher(self.goal_topic, PoseWithCovarianceStamped, reliable_qos)
        self.marker_pub = self.create_secure_publisher(self.marker_topic, MarkerArray, reliable_qos
        )

        # ------------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------------
        self.create_timer(1.0 / self.update_rate, self._explore)

        self.get_logger().info(
            f'FrontierExplorer ready\n'
            f'  map   ← {self.map_topic}\n'
            f'  pose  ← {self.pose_topic}\n'
            f'  goal  → {self.goal_topic}\n'
            f'  min_cluster_size  = {self.min_cluster_size}\n'
            f'  max_frontier_dist = {self.max_frontier_dist} m\n'
            f'  update_rate       = {self.update_rate} Hz\n'
            f'  w_dist={self.w_dist}  w_size={self.w_size}'
        )

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def _map_callback(self, msg: OccupancyGrid):
        self.map_resolution = msg.info.resolution
        self.map_origin_x   = msg.info.origin.position.x
        self.map_origin_y   = msg.info.origin.position.y
        self.map_width      = msg.info.width
        self.map_height     = msg.info.height
        self.map_data       = np.array(msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width))
        self.map_received   = True

    def _drone_map_callback(self, msg: OccupancyGrid):
        """Store the drone/full map for use in _safe_goal validation."""
        self.drone_map_data    = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        self.drone_map_received = True

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x       = msg.pose.pose.position.x
        self.robot_y       = msg.pose.pose.position.y
        self.robot_yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose_received = True


    def _odom_callback(self, msg: Odometry):
        """Real robot mode — transforms EKF odom pose into world frame."""
        pose_odom = PoseStamped()
        pose_odom.header = msg.header    # frame_id = 'odom'
        pose_odom.pose   = msg.pose.pose

        try:
            pose_world = self._tf_buffer.transform(
                pose_odom,
                self.world_frame,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF odom→{self.world_frame} unavailable: {e}',
                throttle_duration_sec=2.0)
            return

        self.robot_x       = pose_world.pose.position.x
        self.robot_y       = pose_world.pose.position.y
        self.robot_yaw = self._yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose_received = True


    def _active_callback(self, msg: Bool):
        self.active = msg.data
        # Clear visited goal blacklist
        if self.active:
            self.visited_goals          = []
            self.reached_goals          = []   # list of (wx, wy, timestamp)
            self.current_goal_x         = None
            self.current_goal_y         = None
            self._last_published_goal   = None
            
        self.get_logger().info(
            f'FrontierExplorer {"activated" if self.active else "deactivated"} '
            f'by MissionController.')
        
    
    def _goal_failed_callback(self, msg: PoseWithCovarianceStamped):
        fx = msg.pose.pose.position.x
        fy = msg.pose.pose.position.y
        self.get_logger().warn(
            f'Goal ({fx:.2f}, {fy:.2f}) is unreachable — '
            f'blacklisting and replanning.')
        # Blacklist the failed goal with a larger radius so we don't keep
        # trying positions in the same lethal area
        self.visited_goals.append((fx, fy))
        self.current_goal_x = None
        self.current_goal_y = None
        self._last_published_goal = None

    def _fov_map_callback(self, msg: OccupancyGrid):
        """Receive the camera-seen mask from camera_fov_tracker."""
        self.fov_map_resolution = msg.info.resolution
        self.fov_map_origin_x   = msg.info.origin.position.x
        self.fov_map_origin_y   = msg.info.origin.position.y
        self.fov_map_width      = msg.info.width
        self.fov_map_height     = msg.info.height
        # Values are 0 (unseen) or 100 (seen) — convert to bool mask
        self.fov_map_data = (
            np.array(msg.data, dtype=np.int8)
              .reshape((self.fov_map_height, self.fov_map_width)) > 0
        )

    def _camera_saw_cell(self, wx: float, wy: float) -> bool:
        """
        Return True if the map cell at world position (wx, wy) has been
        seen by the RGB camera at least once.

        If the fov_map hasn't arrived yet (camera_fov_tracker not running,
        or require_camera_coverage=False), always returns True so exploration
        is not blocked.
        """
        if not self.require_camera_coverage:
            return True
        if self.fov_map_data is None:
            return True   # no mask yet — don't block exploration
        ci = int((wx - self.fov_map_origin_x) / self.fov_map_resolution)
        cj = int((wy - self.fov_map_origin_y) / self.fov_map_resolution)
        if not (0 <= ci < self.fov_map_width and 0 <= cj < self.fov_map_height):
            return False  # outside the tracked area — treat as unseen
        return bool(self.fov_map_data[cj, ci])

    def _cluster_has_unseen_cells(self, cells: list) -> bool:
        """
        Return True if the unknown space BEHIND this frontier cluster has not
        been seen by the camera — i.e. there is unexplored territory adjacent
        to these frontier cells that the camera has never swept.

        Frontier cells are FREE cells bordering UNKNOWN cells. We check the
        unknown neighbours: if any unknown neighbour cell is camera-unseen,
        the robot needs to go there and sweep its camera over it.

        Falls back to True (don't block) if coverage data isn't available.
        """
        if not self.require_camera_coverage:
            return True
        if self.fov_map_data is None:
            return True
        if self.map_data is None:
            return True

        res = self.fov_map_resolution
        ox  = self.fov_map_origin_x
        oy  = self.fov_map_origin_y
        W   = self.fov_map_width
        H   = self.fov_map_height

        cell_set = set(cells)

        for (ci_slam, cj_slam) in cells:
            # Check the 4-connected unknown neighbours of each frontier cell
            for dci, dcj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = ci_slam + dci, cj_slam + dcj
                if not (0 <= ni < self.map_width and 0 <= nj < self.map_height):
                    continue
                if self.map_data[nj, ni] != -1:
                    continue   # not unknown — skip

                # This neighbour is unknown — has the camera seen it?
                wx = self.map_origin_x + (ni + 0.5) * self.map_resolution
                wy = self.map_origin_y + (nj + 0.5) * self.map_resolution
                ci = int((wx - ox) / res)
                cj = int((wy - oy) / res)
                if not (0 <= ci < W and 0 <= cj < H):
                    return True   # outside fov map → treat as unseen
                if not self.fov_map_data[cj, ci]:
                    return True   # unknown neighbour not yet seen by camera

        return False  # all unknown neighbours already seen by camera

    # ==========================================================================
    # Main exploration loop
    # ==========================================================================

    def _explore(self):
        if not self.active:
            return
        if not self.map_received:
            self.get_logger().warn(
                'Waiting for SLAM map…', throttle_duration_sec=5.0)
            return
        # Has to receive an initial pose? Maybe override with default to 0,0
        if not self.pose_received:
            self.get_logger().warn(
                'Waiting for robot pose…', throttle_duration_sec=5.0)
            return
        
        if self.current_goal_x is not None:
            dist_to_goal = math.hypot(
                self.robot_x - self.current_goal_x,
                self.robot_y - self.current_goal_y)
            if dist_to_goal > self.goal_reached_dist:
                self.get_logger().info(
                    f'Moving to frontier ({self.current_goal_x:.2f}, '
                    f'{self.current_goal_y:.2f}) — '
                    f'{dist_to_goal:.2f} m remaining.',
                    throttle_duration_sec=2.0)
                # Re-publish periodically so A* recovers if it restarts mid-journey.
                # Do NOT re-score or pick a new frontier — we are committed.
                if (self._last_published_goal is None or
                        not math.isclose(self.current_goal_x,
                                        self._last_published_goal[0], abs_tol=0.01) or
                        not math.isclose(self.current_goal_y,
                                        self._last_published_goal[1], abs_tol=0.01)):
                    self._last_published_goal = (self.current_goal_x, self.current_goal_y)
                    self._publish_goal(self.current_goal_x, self.current_goal_y)
                return   # ← hard return: no re-scoring, no goal switching
            else:
                # Robot arrived — log it and clear so we pick the next frontier
                self.reached_goals.append((
                    self.current_goal_x,
                    self.current_goal_y,
                    self.get_clock().now().nanoseconds * 1e-9
                ))
                self.get_logger().info(
                    f'Arrived at frontier ({self.current_goal_x:.2f}, '
                    f'{self.current_goal_y:.2f}) — selecting next.')
                self.current_goal_x = None
                self.current_goal_y = None
                self._last_published_goal = None

        # Recompute frontiers
        frontier_cells = self._detect_frontiers()

        if not frontier_cells:
            self.get_logger().info(
                'No frontiers found — map may be fully explored.',
                throttle_duration_sec=5.0)
            return

        clusters = self._cluster_frontiers(frontier_cells)

        if not clusters:
            self.get_logger().info(
                'All clusters below min_cluster_size — nothing to explore.',
                throttle_duration_sec=5.0)
            return

        best = self._pick_best_cluster(clusters)

        if best is None:
            self.get_logger().info(
                'No reachable frontier within max_frontier_dist.',
                throttle_duration_sec=5.0)
            return

        gx, gy = best
        self.current_goal_x = gx
        self.current_goal_y = gy
        # Do NOT blacklist here — only blacklist on _goal_failed_callback.
        # Pre-blacklisting caused valid frontier clusters to be permanently
        # excluded as the robot moves and centroids shift slightly.

        # Only publish if this is a new goal — prevents A* spam
        if (self._last_published_goal is None or
                not math.isclose(gx, self._last_published_goal[0], abs_tol=0.01) or
                not math.isclose(gy, self._last_published_goal[1], abs_tol=0.01)):
            self._last_published_goal = (gx, gy)
            self._publish_goal(gx, gy)
            self.get_logger().info(
                f'Frontier goal → ({gx:.2f}, {gy:.2f}) | '
                f'{len(clusters)} clusters found')

        self._publish_markers(clusters, best)

    # ==========================================================================
    # Step 1 — Detect frontier cells
    # ==========================================================================

    def _detect_frontiers(self) -> list:
        """
        Returns a list of (ci, cj) cell indices where the cell is FREE (0)
        and has at least one UNKNOWN (-1) 4-connected neighbour.

        Using 4-connectivity for the unknown-neighbour check keeps frontiers
        tight to the actual explored boundary and avoids diagonal artefacts.
        """
        grid = self.map_data

        # Vectorised approach using numpy boolean masks and shifts
        # A cell is free if its value is exactly 0 (not unknown, not occupied)
        free_mask = (grid == FREE)

        # Shift the grid in each of the 4 cardinal directions and check for -1
        # np.roll wraps around — mask the border explicitly
        h, w = grid.shape

        has_unknown_neighbour = np.zeros((h, w), dtype=bool)

        for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
            shifted = np.roll(grid, shift, axis=axis)
            # Zero out the wrapped border row/col
            if axis == 0:
                if shift == 1:
                    shifted[0, :] = FREE   # row 0 has no real neighbour above
                else:
                    shifted[-1, :] = FREE
            else:
                if shift == 1:
                    shifted[:, 0] = FREE
                else:
                    shifted[:, -1] = FREE
            has_unknown_neighbour |= (shifted == UNKNOWN)

        frontier_mask = free_mask & has_unknown_neighbour

        # Convert mask to list of (ci, cj) = (col, row)
        rows, cols = np.where(frontier_mask)
        return list(zip(cols.tolist(), rows.tolist()))   # (ci, cj)

    # ==========================================================================
    # Step 2 — Cluster frontier cells (union-find)
    # ==========================================================================

    def _cluster_frontiers(self, frontier_cells: list) -> list:
        """
        Groups frontier cells into connected components using a fast
        union-find over 8-connected adjacency.

        Returns a list of clusters, each being a list of (ci, cj) pairs,
        filtered to only include clusters with >= min_cluster_size cells.
        """
        if not frontier_cells:
            return []

        # Build a set for O(1) membership tests
        cell_set = set(frontier_cells)

        parent = {cell: cell for cell in frontier_cells}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 8-connected neighbours
        for (ci, cj) in frontier_cells:
            for dci, dcj in [( 1, 0),(-1, 0),( 0, 1),( 0,-1),
                              ( 1, 1),( 1,-1),(-1, 1),(-1,-1)]:
                nb = (ci + dci, cj + dcj)
                if nb in cell_set:
                    union((ci, cj), nb)

        # Group by root
        groups: dict = {}
        for cell in frontier_cells:
            root = find(cell)
            groups.setdefault(root, []).append(cell)

        # Filter by minimum size
        return [cells for cells in groups.values()
                if len(cells) >= self.min_cluster_size]

    # ==========================================================================
    # Step 3 — Pick best cluster
    # ==========================================================================

    def _pick_best_cluster(self, clusters: list):
        """
        Scores each cluster with:
          perceived_dist = real_dist / max(heading_score, 0.05)
          score = w_dist * norm_prox + w_size * norm_area

        heading_score is a cosine-based term in [0, 1]:
          0°   → 1.0  (straight ahead — no inflation)
          90°  → 0.5  (side — appears twice as far)
          150° → 0.03 (near-behind — appears ~33× as far)
          >150°→ multiplied by 0.1 first (hard U-turn deterrent)

        Baking heading into the effective distance means a closer frontier
        behind the robot can never rescue itself with proximity alone —
        it has to compete on equal perceived-distance terms with a farther
        frontier that's straight ahead.  Normalising a separate heading
        term would wash out the penalty whenever all candidates face similar
        directions (range collapses → heading contributes nothing).

        w_heading is intentionally removed from the formula; the parameter
        is kept for launch-file compatibility but ignored.
        """
        robot_x, robot_y = self.robot_x, self.robot_y
        res = self.map_resolution

        self.get_logger().warn(
            f'Scoring {len(clusters)} clusters | '
            f'drone_map_ready={self.drone_map_received} | '
            f'fov_map_ready={self.fov_map_data is not None} | '
            f'require_coverage={self.require_camera_coverage}')

        candidates = []

        for cells in clusters:
            ci_safe, cj_safe = self._safe_centroid(cells)
            wx = self.map_origin_x + (ci_safe + 0.5) * res
            wy = self.map_origin_y + (cj_safe + 0.5) * res

            safe = self._safe_goal(wx, wy)
            if safe is None:
                self.get_logger().warn(
                    f'  Cluster centroid ({wx:.2f},{wy:.2f}) — '
                    f'_safe_goal returned None (lethal or out of bounds)')
                continue
            wx, wy = safe

            dist = math.hypot(wx - robot_x, wy - robot_y)
            if dist > self.max_frontier_dist:
                self.get_logger().warn(
                    f'  Cluster ({wx:.2f},{wy:.2f}) — '
                    f'dist {dist:.2f} > max {self.max_frontier_dist}')
                continue

            if self._is_blacklisted(wx, wy):
                self.get_logger().warn(
                    f'  Cluster ({wx:.2f},{wy:.2f}) — blacklisted')
                continue

            if self._is_recently_reached(wx, wy):
                self.get_logger().warn(
                    f'  Cluster ({wx:.2f},{wy:.2f}) — recently reached, skipping')
                continue

            # Filter micro goals
            if dist < self.min_goal_dist:
                self.get_logger().warn(
                    f'  Cluster ({wx:.2f},{wy:.2f}) — '
                    f'too close ({dist:.2f} m < min {self.min_goal_dist:.2f} m), skipping')
                continue

            if not self._cluster_has_unseen_cells(cells):
                self.get_logger().warn(
                    f'  Cluster ({wx:.2f},{wy:.2f}) — '
                    f'all unknown neighbours already camera-seen')
                continue

            area_m2 = len(cells) * (res ** 2)

            # ── Heading-inflated perceived distance ────────────────────────
            # heading_score ∈ [0, 1]: 1.0 = straight ahead, 0.0 = directly behind.
            bearing    = math.atan2(wy - robot_y, wx - robot_x)
            angle_diff = abs(math.atan2(
                math.sin(bearing - self.robot_yaw),
                math.cos(bearing - self.robot_yaw)
            ))  # in [0, π]

            heading_score = (1.0 + math.cos(angle_diff)) / 2.0

            # Hard penalty for U-turns (>150°): multiplied before dividing distance
            if angle_diff > math.radians(150):
                heading_score *= 0.1

            # Floor avoids division by zero for pure U-turns (heading_score ≈ 0)
            _HEADING_FLOOR = 0.05
            perceived_dist = dist / max(heading_score, _HEADING_FLOOR)

            candidates.append((dist, perceived_dist, area_m2, heading_score, wx, wy))

        if not candidates:
            return None

        # ── Normalise perceived distance and area to [0, 1] ───────────────
        perc_dists = [c[1] for c in candidates]
        areas      = [c[2] for c in candidates]

        min_pd, max_pd = min(perc_dists), max(perc_dists)
        min_a,  max_a  = min(areas),      max(areas)

        range_pd = max_pd - min_pd if max_pd > min_pd else 1.0
        range_a  = max_a  - min_a  if max_a  > min_a  else 1.0

        best_score    = -float('inf')
        best_centroid = None

        for (dist, perceived_dist, area_m2, heading_score, wx, wy) in candidates:
            norm_prox = 1.0 - (perceived_dist - min_pd) / range_pd
            norm_area = (area_m2 - min_a) / range_a

            score = self.w_dist * norm_prox + self.w_size * norm_area

            self.get_logger().debug(
                f'Frontier ({wx:.2f},{wy:.2f}) | dist={dist:.2f}m '
                f'angle={math.degrees(angle_diff):.1f}° '
                f'h={heading_score:.3f} perc_d={perceived_dist:.2f}m | '
                f'score={score:.3f}')

            if score > best_score:
                best_score    = score
                best_centroid = (wx, wy)

        return best_centroid

    # ==========================================================================
    # Safe centroid — snaps to nearest real frontier cell if needed
    # ==========================================================================

    def _safe_centroid(self, cells: list) -> tuple:
        """
        Computes the mean (ci, cj) of the cluster.
        If the rounded integer cell is free and inside the map, returns it.
        Otherwise snaps to the frontier cell closest to the raw mean.
        """
        mean_ci = sum(c[0] for c in cells) / len(cells)
        mean_cj = sum(c[1] for c in cells) / len(cells)

        ci_int = int(round(mean_ci))
        cj_int = int(round(mean_cj))

        if (0 <= ci_int < self.map_width and
                0 <= cj_int < self.map_height and
                self.map_data[cj_int, ci_int] == FREE):
            return ci_int, cj_int

        # Snap: pick the frontier cell geometrically closest to the raw mean
        best = min(cells,
                   key=lambda c: (c[0] - mean_ci) ** 2 + (c[1] - mean_cj) ** 2)
        return best
    

    def _safe_goal(self, wx: float, wy: float) -> tuple | None:
        """
        Walks from the frontier centroid toward the robot, step by step.
        Validates against the DRONE map (same map A* uses) so that goals
        that pass this check will also pass A*'s lethal cell check.
        Falls back to SLAM map if drone map hasn't arrived yet.
        """
        robot_x, robot_y = self.robot_x, self.robot_y
        step = self.map_resolution * 2.0

        clear_radius_cells = max(1, int(self.safe_goal_radius / self.map_resolution))

        # Use drone map for validation if available — matches what A* sees
        check_map = self.drone_map_data if self.drone_map_received else self.map_data

        dx = robot_x - wx
        dy = robot_y - wy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None

        ux = dx / dist
        uy = dy / dist

        for i in range(40):
            cx = wx + ux * step * i
            cy = wy + uy * step * i

            ci = int((cx - self.map_origin_x) / self.map_resolution)
            cj = int((cy - self.map_origin_y) / self.map_resolution)

            if not (0 <= ci < self.map_width and 0 <= cj < self.map_height):
                continue

            # Must be free in SLAM map (known free space)
            if int(self.map_data[cj, ci]) != FREE:
                continue

            # Must be clear in drone map (no lethal inflation)
            obstacle_nearby = False
            for dj in range(-clear_radius_cells, clear_radius_cells + 1):
                for di in range(-clear_radius_cells, clear_radius_cells + 1):
                    ni = ci + di
                    nj = cj + dj
                    if not (0 <= ni < self.map_width and 0 <= nj < self.map_height):
                        continue
                    if int(check_map[nj, ni]) >= LETHAL_THRESHOLD:
                        obstacle_nearby = True
                        break
                if obstacle_nearby:
                    break

            if not obstacle_nearby:
                return cx, cy

        return None
    

    def _is_blacklisted(self, wx: float, wy: float) -> bool:
        for (gx, gy) in self.visited_goals:
            if math.hypot(wx - gx, wy - gy) < self.goal_blacklist_radius:
                return True
        return False

    def _is_recently_reached(self, wx: float, wy: float) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        # Expire old entries first
        self.reached_goals = [
            (gx, gy, t) for (gx, gy, t) in self.reached_goals
            if now - t < self.reached_blacklist_ttl
        ]
        for (gx, gy, _t) in self.reached_goals:
            if math.hypot(wx - gx, wy - gy) < self.reached_blacklist_radius:
                return True
        return False

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _publish_goal(self, gx: float, gy: float):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.pose.pose.position.x    = gx
        msg.pose.pose.position.y    = gy
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.w = 1.0
        # Identity covariance — MissionController / AStarPlanner2 don't use it
        self.secure_publish(self.goal_pub, msg)

    def _publish_markers(self, clusters: list, best_centroid: tuple):
        """
        Publishes a MarkerArray for RViz:
          - SPHERE_LIST: all frontier cells (small cyan dots)
          - SPHERE: best goal centroid (large green sphere)
          - SPHERE: each other cluster centroid (small orange sphere)
        """
        now = self.get_clock().now().to_msg()
        array = MarkerArray()
        marker_id = 0

        # --- All frontier cells as small dots ---
        dot = Marker()
        dot.header.stamp    = now
        dot.header.frame_id = self.world_frame
        dot.ns     = 'frontier_cells'
        dot.id     = marker_id; marker_id += 1
        dot.type   = Marker.SPHERE_LIST
        dot.action = Marker.ADD
        dot.scale.x = dot.scale.y = dot.scale.z = self.map_resolution * 1.5
        dot.color   = ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.5)
        dot.pose.orientation.w = 1.0

        from geometry_msgs.msg import Point
        for cells in clusters:
            for (ci, cj) in cells:
                p = Point()
                p.x = self.map_origin_x + (ci + 0.5) * self.map_resolution
                p.y = self.map_origin_y + (cj + 0.5) * self.map_resolution
                p.z = 0.05
                dot.points.append(p)
        array.markers.append(dot)

        # --- Cluster centroids (orange) ---
        for cells in clusters:
            mean_ci = sum(c[0] for c in cells) / len(cells)
            mean_cj = sum(c[1] for c in cells) / len(cells)
            wx = self.map_origin_x + (mean_ci + 0.5) * self.map_resolution
            wy = self.map_origin_y + (mean_cj + 0.5) * self.map_resolution

            is_best = (math.isclose(wx, best_centroid[0], abs_tol=0.01) and
                       math.isclose(wy, best_centroid[1], abs_tol=0.01))

            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = self.world_frame
            m.ns     = 'frontier_centroids'
            m.id     = marker_id; marker_id += 1
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x    = wx
            m.pose.position.y    = wy
            m.pose.position.z    = 0.15
            m.pose.orientation.w = 1.0

            if is_best:
                m.scale.x = m.scale.y = m.scale.z = 0.35
                m.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.20
                m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)

            array.markers.append(m)

        self.secure_publish(self.marker_pub, array)

    # ==========================================================================
    # Public helpers (used by MissionController)
    # ==========================================================================

    def clear_goal(self):
        """Call this when the mission switches to HOMING — discard current goal."""
        self.current_goal_x = None
        self.current_goal_y = None

    def _yaw_from_quaternion(self, q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    

    def _reset_state(self):
        """Full state reset — called on init and whenever active goes False."""
        self.current_goal_x          = None
        self.current_goal_y          = None
        self._last_published_goal    = None
        self.visited_goals           = []
        self.reached_goals           = []
        self._frontier_goal_dirty    = False
        self.get_logger().info('FrontierExplorer state reset.')


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()