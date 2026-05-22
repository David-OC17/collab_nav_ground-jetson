#!/usr/bin/env python3
"""
A* Trajectory Planner Node for ROS2
=====================================
Plans a path on a probabilistic occupancy grid using A* with:
  - 8-connected grid
  - Octile distance heuristic
  - Occupancy costs used directly as A* edge weights
  - Cubic spline smoothing of the raw A* path
  - Replanning on new goal, map update, or robot stuck detection

Subscribes to:
  - /map                        (nav_msgs/OccupancyGrid) — probabilistic occupancy grid
  - /goal_pose                  (geometry_msgs/PoseStamped) — navigation goal

Uses:
  - TF (map → base_link)        — robot position

Publishes:
  - /trajectory_planner/path              (nav_msgs/Path)          — smoothed spline path
  - /trajectory_planner/path_raw          (nav_msgs/Path)          — raw A* path
  - /trajectory_planner/path_markers      (visualization_msgs/MarkerArray) — RViz markers
"""

import math
import heapq
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LETHAL_THRESHOLD  = 90
DIAGONAL_COST     = math.sqrt(2)
STRAIGHT_COST     = 1.0
STUCK_DIST_THRESH = 0.15
STUCK_TIME_THRESH = 5.0


# ==============================================================================
# Cubic spline utilities (pure Python + numpy, no scipy)
# ==============================================================================

# JUST THE TRAJECTORY POSITIONS (NOT VELOCITY)
def _natural_cubic_spline(points: list, num_samples: int) -> list:
    """
    Fit a natural cubic spline through (x,y) points using arc-length
    parameterisation. Returns num_samples evenly-spaced interpolated points.
    """
    if len(points) < 2:
        return points
    if len(points) == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        ts = np.linspace(0, 1, num_samples)
        return [(x0 + t*(x1-x0), y0 + t*(y1-y0)) for t in ts]

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    # Arc-length parameter t ∈ [0, 1]
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    dists = np.where(dists == 0, 1e-9, dists)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    t /= t[-1]

    cx = _solve_natural_spline(t, xs)
    cy = _solve_natural_spline(t, ys)

    t_samples = np.linspace(0.0, 1.0, num_samples)
    return [(_eval_spline(cx, t, ts), _eval_spline(cy, t, ts))
            for ts in t_samples]


# FOR VELOCITY CALCULATION
def _natural_cubic_spline_with_velocity(points, num_samples):
        """
        Returns list of (x, y, vx, vy) — position and velocity at each sample.
        Speed and heading can be derived from vx, vy.
        """
        if len(points) < 2:
            return [(p[0], p[1], 0.0, 0.0) for p in points]

        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        dists = np.where(dists == 0, 1e-9, dists)
        t = np.concatenate([[0.0], np.cumsum(dists)])
        t /= t[-1]

        cx = _solve_natural_spline(t, xs)
        cy = _solve_natural_spline(t, ys)

        t_samples = np.linspace(0.0, 1.0, num_samples)
        result = []
        for ts in t_samples:
            x  = _eval_spline(cx, t, ts)
            y  = _eval_spline(cy, t, ts)
            vx = _eval_spline_derivative(cx, t, ts)
            vy = _eval_spline_derivative(cy, t, ts)
            result.append((x, y, vx, vy))
        return result


def _solve_natural_spline(t: np.ndarray, y: np.ndarray) -> list:
    """Return list of (a,b,c,d) cubic coefficients per segment."""
    n = len(t)
    h = np.diff(t)
    size = n - 2

    if size <= 0:
        slope = (y[1]-y[0]) / (t[1]-t[0]) if (t[1]-t[0]) != 0 else 0.0
        return [(y[0], slope, 0.0, 0.0)]

    diag  = 2.0 * (h[:-1] + h[1:])
    upper = h[1:-1].copy()
    lower = h[1:-1].copy()
    rhs   = 6.0 * ((y[2:]-y[1:-1])/h[1:] - (y[1:-1]-y[:-2])/h[:-1])

    M_inner = _thomas(lower, diag, upper, rhs)
    M = np.concatenate([[0.0], M_inner, [0.0]])

    coeffs = []
    for i in range(n - 1):
        hi = h[i]
        a  = y[i]
        b  = (y[i+1]-y[i])/hi - hi*(2*M[i]+M[i+1])/6.0
        c  = M[i] / 2.0
        d  = (M[i+1]-M[i]) / (6.0*hi)
        coeffs.append((a, b, c, d))
    return coeffs


def _thomas(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal systems."""
    n     = len(diag)
    diag  = diag.copy().astype(float)
    rhs   = rhs.copy().astype(float)
    upper = upper.copy().astype(float)
    lower = lower.copy().astype(float)

    for i in range(1, n):
        f        = lower[i-1] / diag[i-1]
        diag[i] -= f * upper[i-1]
        rhs[i]  -= f * rhs[i-1]

    x = np.zeros(n)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n-2, -1, -1):
        x[i] = (rhs[i] - upper[i]*x[i+1]) / diag[i]
    return x


def _eval_spline(coeffs: list, t_knots: np.ndarray, ts: float) -> float:
    idx = int(np.clip(np.searchsorted(t_knots, ts, side='right')-1,
                      0, len(coeffs)-1))
    dt = ts - t_knots[idx]
    a, b, c, d = coeffs[idx]
    return a + b*dt + c*dt**2 + d*dt**3


# FOR VELOCITY CALCULATION ALONG THE TRAJECTORY
def _eval_spline_derivative(coeffs: list, t_knots: np.ndarray, ts: float) -> float:
    """First derivative of the spline at parameter ts — gives velocity component."""
    idx = int(np.clip(np.searchsorted(t_knots, ts, side='right') - 1,
                      0, len(coeffs) - 1))
    dt = ts - t_knots[idx]
    a, b, c, d = coeffs[idx]
    return b + 2*c*dt + 3*d*dt**2


# ==============================================================================
# Main node
# ==============================================================================

class AStarPlanner(Node):

    def __init__(self):
        super().__init__('astar_planner')

        # /aruco/amr_pose
        # /aruco/goal_pose

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('robot_base_frame',  'base_link')
        self.declare_parameter('map_topic',         '/map')
        self.declare_parameter('goal_topic',        '/goal_pose')
        self.declare_parameter('replan_on_map',     True)
        self.declare_parameter('stuck_detection',   True)
        self.declare_parameter('stuck_check_rate',  1.0)
        self.declare_parameter('path_marker_z',     0.05)
        self.declare_parameter('spline_enabled',    True)
        self.declare_parameter('spline_decimation', 5)    # keep every Nth A* waypoint as knot
        self.declare_parameter('spline_samples',    200)  # output resolution
        self.declare_parameter('inflation_radius',    0.2)
        self.declare_parameter('robot_radius',        0.20)
        self.declare_parameter('cost_scaling',        3.5)

        self.map_frame         = self.get_parameter('map_frame').value
        self.robot_base_frame  = self.get_parameter('robot_base_frame').value
        self.map_topic         = self.get_parameter('map_topic').value
        self.goal_topic        = self.get_parameter('goal_topic').value
        self.replan_on_map     = self.get_parameter('replan_on_map').value
        self.stuck_detection   = self.get_parameter('stuck_detection').value
        self.stuck_check_rate  = self.get_parameter('stuck_check_rate').value
        self.path_marker_z     = self.get_parameter('path_marker_z').value
        self.spline_enabled    = self.get_parameter('spline_enabled').value
        self.spline_decimation = self.get_parameter('spline_decimation').value
        self.spline_samples    = self.get_parameter('spline_samples').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.robot_radius     = self.get_parameter('robot_radius').value
        self.cost_scaling     = self.get_parameter('cost_scaling').value

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.map_data       = None
        self.map_resolution = None
        self.map_origin_x   = None
        self.map_origin_y   = None
        self.map_width      = None
        self.map_height     = None
        self.map_received   = False

        self.goal_x        = None
        self.goal_y        = None
        self.goal_received = False

        self.robot_x = 0.0
        self.robot_y = 0.0

        self.raw_path    = []
        self.smooth_path = []

        self.last_stuck_check_x = None
        self.last_stuck_check_y = None
        self.last_moved_time    = time.time()

        # Inflation
        self.map_resolution   = None   # set on first map callback
        self._inflation_lut   = {}     # built after first map arrives

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------
        # QoS
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

        # ------------------------------------------------------------------
        # Subscribers / Publishers
        # ------------------------------------------------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, map_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self._goal_callback, reliable_qos)

        self.path_pub = self.create_publisher(
            Path, '/trajectory_planner/path', reliable_qos)
        self.raw_path_pub = self.create_publisher(
            Path, '/trajectory_planner/path_raw', reliable_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/trajectory_planner/path_markers', reliable_qos)

        if self.stuck_detection:
            self.stuck_timer = self.create_timer(
                1.0 / self.stuck_check_rate, self._check_stuck)

        self.get_logger().info(
            'AStarPlanner ready | '
            f'map={self.map_topic} | goal={self.goal_topic} | '
            f'spline={self.spline_enabled} '
            f'(decim={self.spline_decimation}, samples={self.spline_samples})'
        )

        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initial_pose_callback,
            10)

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def _map_callback(self, msg: OccupancyGrid):
        self.map_resolution = msg.info.resolution
        self.map_origin_x   = msg.info.origin.position.x
        self.map_origin_y   = msg.info.origin.position.y
        self.map_width      = msg.info.width
        self.map_height     = msg.info.height
        self.map_data       = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))
        self.get_logger().info(f'Map received: {self.map_width}x{self.map_height} cells',throttle_duration_sec=5.0)

        if self.replan_on_map and self.goal_received:
            self._plan()
        
        raw = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))

        # Build LUT once (resolution now known)
        if not self._inflation_lut:
            self._build_inflation_lut()

        # Store inflated map — A* uses this instead of raw
        self.map_data     = self._inflate_map(raw)
        self.map_received = True


    def _initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self.get_logger().info(
            f'Initial pose set: x={self.robot_x:.2f} y={self.robot_y:.2f} yaw={self.robot_yaw:.2f}')


    def _goal_callback(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_received = True
        self.get_logger().info(
            f'New goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        self._plan()
        

    # ==========================================================================
    # Stuck detection
    # ==========================================================================

    def _check_stuck(self):
        if not self.goal_received or not self.smooth_path:
            return
        if not self._update_robot_pose():
            return
        now = time.time()
        if self.last_stuck_check_x is None:
            self.last_stuck_check_x = self.robot_x
            self.last_stuck_check_y = self.robot_y
            self.last_moved_time    = now
            return
        dist = math.hypot(self.robot_x - self.last_stuck_check_x,
                          self.robot_y - self.last_stuck_check_y)
        if dist > STUCK_DIST_THRESH:
            self.last_stuck_check_x = self.robot_x
            self.last_stuck_check_y = self.robot_y
            self.last_moved_time    = now
        elif (now - self.last_moved_time) > STUCK_TIME_THRESH:
            self.get_logger().warn('Robot stuck — replanning')
            self.last_moved_time    = now
            self.last_stuck_check_x = self.robot_x
            self.last_stuck_check_y = self.robot_y
            self._plan()

    # ==========================================================================
    # Planning
    # ==========================================================================

    def _plan(self):
        if not self.map_received:
            self.get_logger().warn('No map yet.')
            return
        if not self._update_robot_pose():
            self.get_logger().warn('No TF pose.')
            return

        sci, scj = self._world_to_cell(self.robot_x, self.robot_y)
        gci, gcj = self._world_to_cell(self.goal_x,  self.goal_y)

        for label, ci, cj in [('Start', sci, scj), ('Goal', gci, gcj)]:
            if not self._in_bounds(ci, cj):
                self.get_logger().error(f'{label} is outside the map.')
                return
        if self._is_lethal(gci, gcj):
            self.get_logger().error('Goal cell is lethal.')
            return

        t0   = time.time()
        path = self._astar(sci, scj, gci, gcj)
        dt   = time.time() - t0

        if path is None:
            self.get_logger().error('A* found no path.')
            self.raw_path = self.smooth_path = []
            return

        self.raw_path = [self._cell_to_world(ci, cj) for ci, cj in path]

        if self.spline_enabled and len(self.raw_path) >= 2:
            self.smooth_path = self._smooth(self.raw_path)
        else:
            self.smooth_path = self.raw_path

        self.get_logger().info(
            f'Path: {len(self.raw_path)} raw → '
            f'{len(self.smooth_path)} spline pts | {dt*1000:.1f} ms')

        self._publish_path(self.smooth_path, self.path_pub)
        self._publish_path(self.raw_path,    self.raw_path_pub)
        self._publish_markers()

    
    def _inflate_map(self, raw: np.ndarray) -> np.ndarray:
        """
        Inflate obstacles in the probabilistic occupancy grid.
        Cost decreases exponentially with distance from the obstacle cell.
        Returns a new float32 array of the same shape.
        """
        inflated = raw.astype(np.float32).copy()
        lethal_cells = np.argwhere(raw >= LETHAL_THRESHOLD)

        for (cj, ci) in lethal_cells:
            for (di, dj), cost in self._inflation_lut.items():
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < self.map_width and 0 <= nj < self.map_height:
                    if inflated[nj, ni] < cost:
                        inflated[nj, ni] = cost

        return inflated

    def _build_inflation_lut(self):
        """
        Precompute for every cell offset within inflation_radius
        what cost it should receive.
        """
        r = int(math.ceil(self.inflation_radius / self.map_resolution))
        inscribed = int(math.ceil(self.robot_radius / self.map_resolution))
        self._inflation_lut = {}

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                dist_cells = math.sqrt(di*di + dj*dj)
                dist_m     = dist_cells * self.map_resolution
                if dist_m > self.inflation_radius:
                    continue
                if dist_cells == 0:
                    continue   # lethal cell itself — already 100
                if dist_cells <= inscribed:
                    cost = 99.0   # inscribed zone — guaranteed collision
                else:
                    cost = 99.0 * math.exp(
                        -self.cost_scaling * (dist_m - self.robot_radius))
                    cost = max(cost, 1.0)
                self._inflation_lut[(di, dj)] = cost

        self.get_logger().info(
            f'Inflation LUT: {len(self._inflation_lut)} cells | '
            f'radius={self.inflation_radius}m | robot_radius={self.robot_radius}m'
        )

    # ==========================================================================
    # Spline smoothing
    # ==========================================================================
    def _smooth(self, raw: list) -> list:
        """
        Decimate A* waypoints to form spline knots, then interpolate a smooth
        natural cubic spline. Decimation removes the 8-connected grid zigzag
        while keeping enough knots to respect obstacle geometry.
        """
        step  = max(1, self.spline_decimation)
        knots = raw[::step]
        if knots[-1] != raw[-1]:
            knots.append(raw[-1])
        if len(knots) < 2:
            return raw
        samples = max(self.spline_samples, len(knots))
        return _natural_cubic_spline_with_velocity(knots, samples)

    # ==========================================================================
    # A*
    # ==========================================================================

    def _astar(self, sci, scj, gci, gcj):
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
                continue
            for dci, dcj, move_cost in neighbours:
                ni, nj = ci+dci, cj+dcj
                if not self._in_bounds(ni, nj) or self._is_lethal(ni, nj):
                    continue
                occ_val  = int(self.map_data[nj, ni])
                occ_cost = 0.5 if occ_val < 0 else (occ_val/100.0)*5.0
                new_g    = g + move_cost + occ_cost
                if new_g < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)] = new_g
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap,
                        (new_g + self._octile(ni, nj, gci, gcj), new_g, ni, nj))
        return None

    def _reconstruct_path(self, came_from, ci, cj):
        path = [(ci, cj)]
        while (ci, cj) in came_from:
            ci, cj = came_from[(ci, cj)]
            path.append((ci, cj))
        path.reverse()
        return path

    def _octile(self, ci, cj, gci, gcj):
        dx, dy = abs(ci-gci), abs(cj-gcj)
        return STRAIGHT_COST*(dx+dy) + (DIAGONAL_COST-2*STRAIGHT_COST)*min(dx,dy)

    # ==========================================================================
    # Publishing
    # ==========================================================================

    # USE THIS WHEN USING TRAJECTORY WITH VELOCITY
    def _publish_path(self, waypoints, publisher):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        if len(waypoints) > 0 and len(waypoints[0]) == 4:
            # Has velocity — normalize and encode into orientation.x/y
            speeds    = [math.hypot(p[2], p[3]) for p in waypoints]
            max_speed = max(speeds) if max(speeds) > 0 else 1.0
            max_robot_speed = 0.5   # ← or make this a parameter
        else:
            max_speed = 1.0
            max_robot_speed = 0.5

        for point in waypoints:
            wx, wy = point[0], point[1]
            vx_world = (point[2] / max_speed) * max_robot_speed if len(point) == 4 else 0.0
            vy_world = (point[3] / max_speed) * max_robot_speed if len(point) == 4 else 0.0

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x    = float(wx)
            pose.pose.position.y    = float(wy)
            pose.pose.position.z    = 0.0
            pose.pose.orientation.x = float(vx_world)
            pose.pose.orientation.y = float(vy_world)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        publisher.publish(msg)


    def _publish_markers(self):
        ma = MarkerArray()

        # Smoothed path — cyan
        if self.smooth_path:
            ma.markers.append(self._line_marker(
                0, self.smooth_path, ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0), 0.04))

        # Raw A* path — semi-transparent yellow
        if self.raw_path:
            ma.markers.append(self._line_marker(
                1, self.raw_path, ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.4), 0.02))

        # Start — green sphere
        ma.markers.append(self._sphere_marker(
            2, self.robot_x, self.robot_y,
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), 0.15))

        # Goal — red-orange sphere
        ma.markers.append(self._sphere_marker(
            3, self.goal_x, self.goal_y,
            ColorRGBA(r=1.0, g=0.2, b=0.0, a=1.0), 0.20))

        # Spline knots — orange dots
        step = max(1, self.spline_decimation)
        for idx in range(0, len(self.raw_path), step):
            wx, wy = self.raw_path[idx][0], self.raw_path[idx][1]
            ma.markers.append(self._sphere_marker(
                100+idx, wx, wy,
                ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9), 0.07))

        self.marker_pub.publish(ma)


    def _line_marker(self, mid, waypoints, color, width):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns      = 'astar_path'
        m.id      = mid
        m.type    = Marker.LINE_STRIP
        m.action  = Marker.ADD
        m.scale.x = width
        m.color   = color
        for point in waypoints:
            wx, wy = point[0], point[1]   # ← fix here
            p = Point()
            p.x = float(wx)
            p.y = float(wy)
            p.z = self.path_marker_z
            m.points.append(p)
        return m


    def _sphere_marker(self, mid, x, y, color, scale):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns     = 'astar_path'
        m.id     = mid
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x    = float(x)
        m.pose.position.y    = float(y)
        m.pose.position.z    = self.path_marker_z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color   = color
        return m

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _update_robot_pose(self) -> bool:
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_base_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            self.robot_x = tf_stamped.transform.translation.x
            self.robot_y = tf_stamped.transform.translation.y
            return True
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF unavailable: {e}', throttle_duration_sec=3.0)
            return False

    def _world_to_cell(self, wx, wy):
        ci = int((wx - self.map_origin_x) / self.map_resolution)
        cj = int((wy - self.map_origin_y) / self.map_resolution)
        return ci, cj

    def _cell_to_world(self, ci, cj):
        wx = self.map_origin_x + (ci + 0.5) * self.map_resolution
        wy = self.map_origin_y + (cj + 0.5) * self.map_resolution
        return wx, wy

    def _in_bounds(self, ci, cj):
        return 0 <= ci < self.map_width and 0 <= cj < self.map_height

    def _is_lethal(self, ci, cj):
        return int(self.map_data[cj, ci]) >= LETHAL_THRESHOLD


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()