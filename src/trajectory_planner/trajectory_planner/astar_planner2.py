#!/usr/bin/env python3
"""
A* Trajectory Planner v2 for ROS2
=====================================
Plans a collision-free path on a fused occupancy grid using A* with:
  - 8-connected grid with octile heuristic
  - Occupancy costs used directly as A* edge weights
  - Obstacle inflation with exponential cost decay
  - Smart replanning: only replan when truly necessary

Subscribes:
  - /fusion/map              (nav_msgs/OccupancyGrid)                  — fused occupancy grid
  - /aruco/goal/pose         (geometry_msgs/PoseWithCovarianceStamped) — navigation goal

Robot pose is obtained exclusively from TF: world → base_footprint.

Publishes:
  - /trajectory_planner2/path  (nav_msgs/Path)  — A* path waypoints

Replanning triggers:
  1. FIRST PLAN — map + goal available and no path exists yet.

  2. NEW GOAL — goal topic receives a pose farther than `goal_change_threshold` metres
     from the last accepted goal. Clears the current path and replans immediately.

  3. MAP UPDATE + PATH BLOCKED — any cell along the current path has inflated cost
     above `collision_cost_threshold` in the new map → replan.

  4. MAP UPDATE + SIGNIFICANT CHANGE — two independent metrics are evaluated;
     EITHER exceeding its threshold triggers a replan:
       a. Global ratio   : newly_occupied_cells / total_cells > global_change_threshold
          (cells that transitioned from free/unknown → occupied)
       b. Proximity score: sum(exp(-d_i / path_proximity_radius)) for every newly
          occupied cell, where d_i is the cell's distance to the nearest path waypoint.
          Score > path_proximity_threshold triggers replan.

All replanning is rate-limited by `min_replan_interval_sec`.

Parameters (all configurable at launch):
  map_topic                 '/fusion/map'
  goal_topic                '/aruco/goal/pose'
  world_frame               'world'
  robot_base_frame          'base_footprint'
  goal_change_threshold     0.30     m     — min goal displacement to trigger replan
  collision_cost_threshold  80.0           — inflated cost above which path is blocked
  global_change_threshold   0.05           — ratio [0,1] of new obstacles in full map
  path_proximity_threshold  5.0            — weighted sum of new obstacles near path
  path_proximity_radius     2.0     m      — decay distance for proximity weighting
  min_replan_interval_sec   3.0     s      — minimum time between replans
  inflation_radius          0.20    m
  robot_radius              0.20    m
  cost_scaling              3.5
"""

import math
import heapq
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import tf2_ros
from tf2_ros import TransformException

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from ros2_security import SecureNodeMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LETHAL_THRESHOLD = 90
DIAGONAL_COST    = math.sqrt(2)
STRAIGHT_COST    = 1.0


# ==============================================================================
# Main node
# ==============================================================================

class AStarPlanner2(SecureNodeMixin, Node):

    def __init__(self):
        super().__init__('astar_planner2')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        # Topics
        self.declare_parameter('map_topic',        '/drone/map')
        self.declare_parameter('goal_topic',       '/aruco/goal/pose')
        self.declare_parameter('world_frame',      'world')
        self.declare_parameter('robot_base_frame', 'base_footprint')

        # Goal change detection
        self.declare_parameter('goal_change_threshold', 0.30)   # metres

        # Path collision detection (Stage 1)
        self.declare_parameter('collision_cost_threshold', 80.0)  # inflated cost

        # Map diff detection (Stage 2)
        self.declare_parameter('global_change_threshold',   0.05)  # ratio [0, 1]
        self.declare_parameter('path_proximity_threshold',  5.0)   # weighted sum
        self.declare_parameter('path_proximity_radius',     2.0)   # metres

        # Rate limiting
        self.declare_parameter('min_replan_interval_sec', 3.0)

        # Inflation
        self.declare_parameter('inflation_radius', 0.20)
        self.declare_parameter('robot_radius',     0.20)
        self.declare_parameter('cost_scaling',     3.5)

        # Read all params
        self.map_topic        = self.get_parameter('map_topic').value
        self.goal_topic       = self.get_parameter('goal_topic').value
        self.world_frame      = self.get_parameter('world_frame').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value

        self.goal_change_threshold    = float(self.get_parameter('goal_change_threshold').value)
        self.collision_cost_threshold = float(self.get_parameter('collision_cost_threshold').value)
        self.global_change_threshold  = float(self.get_parameter('global_change_threshold').value)
        self.path_proximity_threshold = float(self.get_parameter('path_proximity_threshold').value)
        self.path_proximity_radius    = float(self.get_parameter('path_proximity_radius').value)
        self.min_replan_interval_sec  = float(self.get_parameter('min_replan_interval_sec').value)
        self.inflation_radius         = float(self.get_parameter('inflation_radius').value)
        self.robot_radius             = float(self.get_parameter('robot_radius').value)
        self.cost_scaling             = float(self.get_parameter('cost_scaling').value)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        # Map
        self.map_data       = None    # current inflated map (float32 ndarray[height, width])
        self.prev_raw_map   = None    # last raw map, for diff computation
        self.map_resolution = None
        self.map_origin_x   = None
        self.map_origin_y   = None
        self.map_width      = None
        self.map_height     = None
        self.map_received   = False
        self._inflation_lut = {}      # {(di, dj): cost} — built once after first map

        # Robot pose (read from TF at plan time)
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.tf_ready  = False    # True once world→base_footprint is available

        # Goal
        self.goal_x        = None
        self.goal_y        = None
        self.last_goal_x   = None
        self.last_goal_y   = None
        self.goal_received = False

        # Active path
        self.current_path = []

        # Rate limiting
        self.last_replan_time = 0.0

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.create_timer(1.0, self._tf_probe)

        # ------------------------------------------------------------------
        # QoS profiles
        # ------------------------------------------------------------------
        map_qos = QoSProfile(
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
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.map_sub  = self.create_secure_subscription(self.map_topic, OccupancyGrid, self._map_callback, min_level=None, qos=map_qos)
        self.goal_sub = self.create_secure_subscription(self.goal_topic, PoseWithCovarianceStamped, self._goal_callback, min_level=None, qos=latched_qos)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.path_pub = self.create_secure_publisher('/trajectory_planner2/path', Path, reliable_qos)

        # Notify when goal is lethal and planner failed (to replan in frontier explorer)
        self.goal_failed_pub = self.create_secure_publisher('/astar/goal_failed', PoseWithCovarianceStamped, reliable_qos)

        self.get_logger().info(
            'AStarPlanner2 ready\n'
            f'  map   → {self.map_topic}\n'
            f'  goal  → {self.goal_topic}\n'
            f'  pose  ← TF {self.world_frame} → {self.robot_base_frame}\n'
            f'  out   → /trajectory_planner2/path\n'
            f'  collision_cost_threshold  = {self.collision_cost_threshold}\n'
            f'  global_change_threshold   = {self.global_change_threshold}\n'
            f'  path_proximity_threshold  = {self.path_proximity_threshold}\n'
            f'  path_proximity_radius     = {self.path_proximity_radius} m\n'
            f'  goal_change_threshold     = {self.goal_change_threshold} m'
        )

    # ==========================================================================
    # TF probe — 1 Hz, fires a first plan once TF becomes available
    # ==========================================================================

    def _tf_probe(self):
        if self.tf_ready:
            return
        try:
            self._tf_buffer.lookup_transform(
                self.world_frame, self.robot_base_frame, rclpy.time.Time())
        except TransformException:
            self.get_logger().warn(
                f'Waiting for TF {self.world_frame} → {self.robot_base_frame}…',
                throttle_duration_sec=5.0)
            return

        self.tf_ready = True
        self.get_logger().info(
            f'TF {self.world_frame} → {self.robot_base_frame} available — ready to plan.')

        if self.map_received and self.goal_received and not self.current_path:
            self._rate_limited_plan('first plan after TF became available')

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def _map_callback(self, msg: OccupancyGrid):

        w   = msg.info.width
        h   = msg.info.height
        res = msg.info.resolution

        new_raw = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # Update grid metadata (may change if map grows)
        self.map_resolution = res
        self.map_origin_x   = msg.info.origin.position.x
        self.map_origin_y   = msg.info.origin.position.y
        self.map_width      = w
        self.map_height     = h

        # Build inflation LUT on first call (needs map_resolution)
        if not self._inflation_lut:
            self._build_inflation_lut()

        new_inflated = self._inflate_map(new_raw)

        # ── FIRST MAP ──────────────────────────────────────────────────────
        if not self.map_received:
            self.map_data     = new_inflated
            self.prev_raw_map = new_raw.copy()
            self.map_received = True
            self.get_logger().info(
                f'First map received: {w}×{h} cells @ {res:.3f} m/cell')
            if self.tf_ready and self.goal_received:
                self._rate_limited_plan('first plan after map arrived')
            return

        # ── SUBSEQUENT MAP — NO ACTIVE PATH ───────────────────────────────
        if not self.current_path:
            self.map_data     = new_inflated
            self.prev_raw_map = new_raw.copy()
            if self.tf_ready and self.goal_received:
                self._rate_limited_plan('no active path — planning after map update')
            return

        # ── SUBSEQUENT MAP — ACTIVE PATH EXISTS ───────────────────────────
        should_replan = False
        reason        = ''

        # Stage 1 — Path collision check
        if self._is_path_blocked(new_inflated):
            should_replan = True
            reason = (f'path blocked: at least one waypoint has '
                      f'inflated cost > {self.collision_cost_threshold}')

        # Stage 2 — Map-diff metrics (only if Stage 1 did not already trigger)
        if not should_replan and self.prev_raw_map is not None:
            if self.prev_raw_map.shape == new_raw.shape:
                global_ratio, prox_score = self._map_diff_scores(
                    self.prev_raw_map, new_raw)

                self.get_logger().debug(
                    f'Map diff — global_ratio={global_ratio:.4f} '
                    f'prox_score={prox_score:.2f}',
                    throttle_duration_sec=5.0)

                if global_ratio > self.global_change_threshold:
                    should_replan = True
                    reason = (f'global map change ratio {global_ratio:.4f} '
                              f'> threshold {self.global_change_threshold:.4f}')
                elif prox_score > self.path_proximity_threshold:
                    should_replan = True
                    reason = (f'proximity-weighted change score {prox_score:.2f} '
                              f'> threshold {self.path_proximity_threshold:.2f}')
            else:
                # Map resized — always safer to replan
                should_replan = True
                reason = (f'map dimensions changed '
                          f'{self.prev_raw_map.shape} → {new_raw.shape}')

        # Commit new map regardless of replan decision
        self.map_data     = new_inflated
        self.prev_raw_map = new_raw.copy()

        if should_replan:
            self.get_logger().info(f'Map update → replanning ({reason})')
            self._rate_limited_plan(reason)
        else:
            self.get_logger().debug(
                'Map update: path unaffected — no replan.',
                throttle_duration_sec=5.0)

    def _goal_callback(self, msg: PoseWithCovarianceStamped):

        new_gx = msg.pose.pose.position.x
        new_gy = msg.pose.pose.position.y

        # Determine whether this goal is meaningfully different
        if self.last_goal_x is None:
            goal_is_new = True
        else:
            dist = math.hypot(new_gx - self.last_goal_x, new_gy - self.last_goal_y)
            goal_is_new = dist > self.goal_change_threshold

        # Always update internal goal coordinates
        self.goal_x        = new_gx
        self.goal_y        = new_gy
        self.goal_received = True

        if not goal_is_new and self.current_path:
            self.get_logger().debug(
                f'Goal update within {self.goal_change_threshold} m — keeping path.',
                throttle_duration_sec=3.0)
            return

        # Commit as new accepted goal and invalidate stale path
        self.last_goal_x  = new_gx
        self.last_goal_y  = new_gy
        self.current_path = []

        self.get_logger().info(
            f'New goal accepted: ({new_gx:.2f}, {new_gy:.2f})')

        if self.tf_ready and self.map_received:
            self._rate_limited_plan('new goal received')

    # ==========================================================================
    # Rate-limited planning
    # ==========================================================================

    def _rate_limited_plan(self, reason: str = ''):
        """Calls _plan() only if min_replan_interval_sec has elapsed since last replan."""
        now     = time.time()
        elapsed = now - self.last_replan_time
        if elapsed < self.min_replan_interval_sec:
            self.get_logger().debug(
                f'Replan suppressed ({reason}) — '
                f'{elapsed:.1f}s elapsed, min={self.min_replan_interval_sec}s.')
            return
        self.last_replan_time = now
        self._plan()

    # ==========================================================================
    # A* Planning
    # ==========================================================================

    def _plan(self):

        # Guard: all required data must be present
        if not self.tf_ready:
            self.get_logger().warn('Plan requested — waiting for TF.')
            return
        if not self.map_received:
            self.get_logger().warn('Plan requested — waiting for map.')
            return
        if not self.goal_received:
            self.get_logger().warn('Plan requested — waiting for goal.')
            return

        try:
            tf = self._tf_buffer.lookup_transform(
                self.world_frame, self.robot_base_frame, rclpy.time.Time())
            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y
        except TransformException as e:
            self.get_logger().warn(f'Cannot get robot pose from TF: {e}')
            return

        self.get_logger().info(
            f'Planning: robot=({robot_x:.2f},{robot_y:.2f}) '
            f'→ goal=({self.goal_x:.2f},{self.goal_y:.2f})')

        sci, scj = self._world_to_cell(robot_x, robot_y)
        gci, gcj = self._world_to_cell(self.goal_x, self.goal_y)

        # Bounds check
        for label, ci, cj in [('Start', sci, scj), ('Goal', gci, gcj)]:
            if not self._in_bounds(ci, cj):
                self.get_logger().error(
                    f'{label} cell ({ci},{cj}) is outside the map '
                    f'({self.map_width}×{self.map_height}).')
                return

        # Goal lethality check
        if self._is_lethal(gci, gcj):
            # Notify frontier explorer so it can blacklist and pick a new goal
            fail_msg = PoseWithCovarianceStamped()
            fail_msg.header.stamp    = self.get_clock().now().to_msg()
            fail_msg.header.frame_id = self.world_frame
            fail_msg.pose.pose.position.x = self.goal_x
            fail_msg.pose.pose.position.y = self.goal_y
            self.secure_publish(self.goal_failed_pub, fail_msg)

            self.get_logger().error(
                f'Goal cell ({gci},{gcj}) is lethal — cannot plan.')
            return

        t0   = time.time()
        path = self._astar(sci, scj, gci, gcj)
        dt   = time.time() - t0

        if path is None:
            self.get_logger().error(
                f'A* found no path after {dt*1000:.1f} ms — '
                'goal may be surrounded by obstacles.')
            self.current_path = []
            return

        self.current_path = [self._cell_to_world(ci, cj) for ci, cj in path]

        self.get_logger().info(
            f'Path found: {len(self.current_path)} waypoints | {dt*1000:.1f} ms')

        self._publish_path(self.current_path)

    # ==========================================================================
    # Stage 1 — Path collision check
    # ==========================================================================

    def _is_path_blocked(self, inflated_map: np.ndarray) -> bool:

        for wx, wy in self.current_path:
            ci, cj = self._world_to_cell(wx, wy)
            if not self._in_bounds(ci, cj):
                continue   # waypoint outside map (can happen after resize)
            if float(inflated_map[cj, ci]) > self.collision_cost_threshold:
                return True
        return False

    # ==========================================================================
    # Stage 2 — Map diff metrics
    # ==========================================================================

    def _map_diff_scores(self, old_raw: np.ndarray, new_raw: np.ndarray):

        total_cells = old_raw.size

        # Newly occupied cells: free/unknown → blocked
        was_free     = old_raw < LETHAL_THRESHOLD   # includes unknown (-1)
        now_blocked  = new_raw >= LETHAL_THRESHOLD
        new_obs_mask = was_free & now_blocked

        new_obs_count = int(np.sum(new_obs_mask))
        global_ratio  = new_obs_count / total_cells if total_cells > 0 else 0.0

        # Proximity-weighted score
        prox_score = 0.0
        if new_obs_count > 0 and self.current_path:
            new_obs_cells = np.argwhere(new_obs_mask)  # shape (N,2): [row=j, col=i]

            # Vectorise path waypoints for fast batch distance computation
            path_wx = np.array([p[0] for p in self.current_path], dtype=np.float32)
            path_wy = np.array([p[1] for p in self.current_path], dtype=np.float32)

            for (nj, ni) in new_obs_cells:
                # Cell centre → world coords
                wx = self.map_origin_x + (float(ni) + 0.5) * self.map_resolution
                wy = self.map_origin_y + (float(nj) + 0.5) * self.map_resolution

                # Min distance to any path waypoint
                dists    = np.hypot(path_wx - wx, path_wy - wy)
                min_dist = float(np.min(dists))

                # Exponential decay: near path → weight ≈ 1, far → weight ≈ 0
                prox_score += math.exp(-min_dist / self.path_proximity_radius)

        return global_ratio, prox_score

    # ==========================================================================
    # Obstacle inflation (identical to astar_planner v1)
    # ==========================================================================

    def _inflate_map(self, raw: np.ndarray) -> np.ndarray:

        inflated     = raw.astype(np.float32).copy()
        lethal_cells = np.argwhere(raw >= LETHAL_THRESHOLD)   # [row=j, col=i]

        for (cj, ci) in lethal_cells:
            for (di, dj), cost in self._inflation_lut.items():
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < self.map_width and 0 <= nj < self.map_height:
                    if inflated[nj, ni] < cost:
                        inflated[nj, ni] = cost
        return inflated

    def _build_inflation_lut(self):

        r         = int(math.ceil(self.inflation_radius / self.map_resolution))
        inscribed = int(math.ceil(self.robot_radius     / self.map_resolution))
        self._inflation_lut = {}

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                dist_cells = math.sqrt(di * di + dj * dj)
                dist_m     = dist_cells * self.map_resolution
                if dist_m > self.inflation_radius or dist_cells == 0:
                    continue
                if dist_cells <= inscribed:
                    cost = 99.0
                else:
                    cost = max(
                        99.0 * math.exp(-self.cost_scaling * (dist_m - self.robot_radius)),
                        1.0)
                self._inflation_lut[(di, dj)] = cost

        self.get_logger().info(
            f'Inflation LUT built: {len(self._inflation_lut)} offsets | '
            f'inflation_radius={self.inflation_radius} m | '
            f'robot_radius={self.robot_radius} m | '
            f'cost_scaling={self.cost_scaling}')

    # ==========================================================================
    # A* core 
    # ==========================================================================

    def _astar(self, sci: int, scj: int, gci: int, gcj: int):

        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, sci, scj))
        came_from = {}
        g_score   = {(sci, scj): 0.0}

        neighbours = [
            ( 1,  0, STRAIGHT_COST), (-1,  0, STRAIGHT_COST),
            ( 0,  1, STRAIGHT_COST), ( 0, -1, STRAIGHT_COST),
            ( 1,  1, DIAGONAL_COST), ( 1, -1, DIAGONAL_COST),
            (-1,  1, DIAGONAL_COST), (-1, -1, DIAGONAL_COST),
        ]

        while open_heap:
            f, g, ci, cj = heapq.heappop(open_heap)

            if (ci, cj) == (gci, gcj):
                return self._reconstruct_path(came_from, ci, cj)

            if g > g_score.get((ci, cj), float('inf')):
                continue   # stale open-set entry

            for dci, dcj, move_cost in neighbours:
                ni, nj = ci + dci, cj + dcj
                if not self._in_bounds(ni, nj) or self._is_lethal(ni, nj):
                    continue

                occ_val  = int(self.map_data[nj, ni])
                occ_cost = 0.5 if occ_val < 0 else (occ_val / 100.0) * 5.0
                new_g    = g + move_cost + occ_cost

                if new_g < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)]   = new_g
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(
                        open_heap,
                        (new_g + self._octile(ni, nj, gci, gcj), new_g, ni, nj))
        return None

    def _reconstruct_path(self, came_from: dict, ci: int, cj: int) -> list:
        path = [(ci, cj)]
        while (ci, cj) in came_from:
            ci, cj = came_from[(ci, cj)]
            path.append((ci, cj))
        path.reverse()
        return path

    def _octile(self, ci: int, cj: int, gci: int, gcj: int) -> float:
        dx, dy = abs(ci - gci), abs(cj - gcj)
        return STRAIGHT_COST * (dx + dy) + (DIAGONAL_COST - 2 * STRAIGHT_COST) * min(dx, dy)

    # ==========================================================================
    # Publishing
    # ==========================================================================

    def _publish_path(self, waypoints: list):
        """Serialises a list of (wx, wy) tuples into nav_msgs/Path and publishes it."""
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame

        for wx, wy in waypoints:
            pose = PoseStamped()
            pose.header             = msg.header
            pose.pose.position.x    = float(wx)
            pose.pose.position.y    = float(wy)
            pose.pose.position.z    = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.secure_publish(self.path_pub, msg)

    # ==========================================================================
    # Grid helpers
    # ==========================================================================

    def _world_to_cell(self, wx: float, wy: float):
        ci = int((wx - self.map_origin_x) / self.map_resolution)
        cj = int((wy - self.map_origin_y) / self.map_resolution)
        return ci, cj

    def _cell_to_world(self, ci: int, cj: int):
        wx = self.map_origin_x + (ci + 0.5) * self.map_resolution
        wy = self.map_origin_y + (cj + 0.5) * self.map_resolution
        return wx, wy

    def _in_bounds(self, ci: int, cj: int) -> bool:
        return 0 <= ci < self.map_width and 0 <= cj < self.map_height

    def _is_lethal(self, ci: int, cj: int) -> bool:
        return float(self.map_data[cj, ci]) >= LETHAL_THRESHOLD


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
