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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FREE     =  0
UNKNOWN  = -1
OCCUPIED = 100
LETHAL_THRESHOLD = 50   # cells above this are treated as occupied


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

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
        self.declare_parameter('goal_reached_dist', 0.40)
        self.declare_parameter('w_dist',            0.7)
        self.declare_parameter('w_size',            0.3)
        self.declare_parameter('active',            True)
        self.declare_parameter('odom_topic', '')   # Switch between simulation and real robot
        self.declare_parameter('safe_goal_radius', 0.30)   # m — must be >= astar inflation_radius

        self.map_topic         = self.get_parameter('map_topic').value
        self.pose_topic        = self.get_parameter('pose_topic').value
        self.goal_topic        = self.get_parameter('goal_topic').value
        self.marker_topic      = self.get_parameter('marker_topic').value
        self.world_frame       = self.get_parameter('world_frame').value
        self.min_cluster_size  = int(self.get_parameter('min_cluster_size').value)
        self.max_frontier_dist = float(self.get_parameter('max_frontier_dist').value)
        self.update_rate       = float(self.get_parameter('update_rate').value)
        self.goal_reached_dist = float(self.get_parameter('goal_reached_dist').value)
        self.w_dist            = float(self.get_parameter('w_dist').value)
        self.w_size            = float(self.get_parameter('w_size').value)
        self.active            = bool(self.get_parameter('active').value)
        self.odom_topic        = self.get_parameter('odom_topic').value
        self.safe_goal_radius = float(self.get_parameter('safe_goal_radius').value)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.map_data       = None   # np.ndarray (height, width) int8
        self.map_resolution = None
        self.map_origin_x   = None
        self.map_origin_y   = None
        self.map_width      = None
        self.map_height     = None
        self.map_received   = False

        self.robot_x        = 0.0
        self.robot_y        = 0.0
        self.pose_received  = False

        self.current_goal_x = None
        self.current_goal_y = None

        self.visited_goals         = []    # list of (wx, wy) already sent
        self.goal_blacklist_radius = 0.50  # m — don't revisit within this radius   

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
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._map_callback,
            latched_qos
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
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/astar/goal_failed',
            self._goal_failed_callback,
            reliable_qos
        )

        # MissionController can silence/activate this node at runtime
        self.active_sub = self.create_subscription(
            Bool,
            '/frontier_explorer/active',
            self._active_callback,
            reliable_qos
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.goal_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.goal_topic,
            reliable_qos
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            reliable_qos
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

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x       = msg.pose.pose.position.x
        self.robot_y       = msg.pose.pose.position.y
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
        self.pose_received = True


    def _active_callback(self, msg: Bool):
        self.active = msg.data
        # Clear visited goal blacklist
        if self.active:
            self.visited_goals  = []
            self.current_goal_x = None
            self.current_goal_y = None
            
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
        # Clear current goal so _explore() picks a new one immediately
        self.current_goal_x = None
        self.current_goal_y = None

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
        
        # ── Wait until robot reaches current goal before picking a new one ──
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
                return   # ← don't pick a new goal yet

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
        # Add to blacklist
        self.visited_goals.append((gx, gy)) 

        self._publish_goal(gx, gy)
        self._publish_markers(clusters, best)

        self.get_logger().info(
            f'Frontier goal → ({gx:.2f}, {gy:.2f}) | '
            f'{len(clusters)} clusters found')

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
          score = w_dist * (1 / distance_m) + w_size * cluster_area_m2

        where:
          distance_m      = Euclidean distance robot → safe centroid
          cluster_area_m2 = len(cells) * resolution²  (information gain proxy)

        Both terms are normalised across all candidates before weighting so
        that neither dominates purely due to scale differences.

        Returns the (world_x, world_y) of the highest-scoring cluster's safe
        centroid, or None if no cluster qualifies within max_frontier_dist.
        """
        robot_x, robot_y = self.robot_x, self.robot_y
        res = self.map_resolution

        # --- Compute raw metrics for every qualifying cluster ---
        candidates = []   # (distance_m, area_m2, wx, wy)

        for cells in clusters:
            ci_safe, cj_safe = self._safe_centroid(cells)
            wx = self.map_origin_x + (ci_safe + 0.5) * res
            wy = self.map_origin_y + (cj_safe + 0.5) * res

            safe = self._safe_goal(wx, wy)
            if safe is None:
                continue       # skip this cluster entirely
            wx, wy = safe

            dist = math.hypot(wx - robot_x, wy - robot_y)
            if dist > self.max_frontier_dist:
                continue

            if self._is_blacklisted(wx, wy):  
                continue

            area_m2 = len(cells) * (res ** 2)
            candidates.append((dist, area_m2, wx, wy))

        if not candidates:
            return None

        # --- Normalise each metric to [0, 1] across candidates ---
        dists   = [c[0] for c in candidates]
        areas   = [c[1] for c in candidates]

        min_d, max_d = min(dists), max(dists)
        min_a, max_a = min(areas), max(areas)

        range_d = max_d - min_d if max_d > min_d else 1.0
        range_a = max_a - min_a if max_a > min_a else 1.0

        # --- Score: higher is better ---
        best_score    = -float('inf')
        best_centroid = None

        for (dist, area_m2, wx, wy) in candidates:
            norm_prox = 1.0 - (dist  - min_d) / range_d   # closer  → higher
            norm_area = (area_m2 - min_a) / range_a        # larger  → higher

            score = self.w_dist * norm_prox + self.w_size * norm_area

            self.get_logger().debug(
                f'Frontier ({wx:.2f},{wy:.2f}) | dist={dist:.2f}m '
                f'area={area_m2:.3f}m² | score={score:.3f}')

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
        Accepts the first cell that is FREE and has no OCCUPIED neighbour
        within inflation_radius cells — approximating the inflated drone map
        without needing direct access to it.
        """
        robot_x, robot_y = self.robot_x, self.robot_y
        step = self.map_resolution * 2.0   # step 2 cells at a time toward robot

        # How many cells to check around the candidate for obstacles
        # Matches AStarPlanner2 default inflation_radius=0.20m / 0.05 res = 4 cells
        # After — use a parameter so it can be tuned to match astar_planner2's inflation
        clear_radius_cells = max(1, int(self.safe_goal_radius / self.map_resolution))

        dx = robot_x - wx
        dy = robot_y - wy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None

        # Unit vector toward robot
        ux = dx / dist
        uy = dy / dist

        # Try up to 20 steps toward robot
        for i in range(20):
            cx = wx + ux * step * i
            cy = wy + uy * step * i

            ci = int((cx - self.map_origin_x) / self.map_resolution)
            cj = int((cy - self.map_origin_y) / self.map_resolution)

            if not (0 <= ci < self.map_width and 0 <= cj < self.map_height):
                continue

            if int(self.map_data[cj, ci]) != FREE:
                continue

            # Must have no OCCUPIED cell within clear_radius_cells
            obstacle_nearby = False
            for dj in range(-clear_radius_cells, clear_radius_cells + 1):
                for di in range(-clear_radius_cells, clear_radius_cells + 1):
                    ni = ci + di
                    nj = cj + dj
                    if not (0 <= ni < self.map_width and 0 <= nj < self.map_height):
                        continue
                    if int(self.map_data[nj, ni]) >= LETHAL_THRESHOLD:
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
        self.goal_pub.publish(msg)

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

        self.marker_pub.publish(array)

    # ==========================================================================
    # Public helpers (used by MissionController)
    # ==========================================================================

    def clear_goal(self):
        """Call this when the mission switches to HOMING — discard current goal."""
        self.current_goal_x = None
        self.current_goal_y = None


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