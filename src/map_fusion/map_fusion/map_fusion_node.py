"""``map_fusion_node`` -- the ROS 2 node that fuses the drone map and the SLAM
Toolbox map.

It owns Stages 3, 6 and 7 of the pipeline directly and delegates Stages 1/2/4/5
to :mod:`preprocessing`, :mod:`coarse_search` and :mod:`icp`.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped)
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from std_msgs.msg import Float32, String
from tf2_ros import (Buffer, StaticTransformBroadcaster, TransformException,
                     TransformListener)

from .coarse_search import coarse_search
from .geometry import (angle_diff, apply_se2, compose, invert,
                       quaternion_from_yaw, yaw_from_quaternion)
from .grid_utils import (array_to_occupancygrid, occupancygrid_to_array,
                         reproject_slam_grid)
from .icp import icp_align, residual_to_confidence
from .preprocessing import preprocess_grid

# (name, default) for every declared parameter. Defaults follow the spec.
PARAM_SPEC = [
    # --- topic / frame names ---
    ('drone_map_topic', '/drone/map'),
    ('slam_map_topic', '/amr/map'),
    ('aruco_pose_topic', '/aruco/amr_pose'),
    ('aruco_tracking_topic', '/aruco/amr_tracking'),
    ('reprojected_topic', '/fusion/slam_reprojected'),
    ('confidence_topic', '/fusion/confidence'),
    ('status_topic', '/fusion/status'),
    ('world_frame', 'world'),
    ('slam_frame', 'slam_map'),
    ('base_frame', 'base_link'),
    ('use_live_tracking', False),
    # --- preprocessing ---
    ('occupied_threshold_drone', 65),
    ('occupied_threshold_slam', 65),
    ('edge_kernel_size', 1),
    ('output_resolution_m', 0.02),
    # --- coarse search ---
    ('coarse_translation_step_m', 0.10),
    ('coarse_rotation_step_deg', 5.0),
    ('coarse_translation_radius_m', 1.0),
    ('coarse_top_k', 5),
    ('symmetry_score_tolerance', 0.05),
    # --- ICP ---
    ('icp_max_correspondence_m', 0.10),
    ('icp_max_iterations', 50),
    ('icp_convergence_epsilon', 1e-4),
    ('icp_residual_scale', 0.05),
    # --- validation / gating ---
    ('confidence_rerun_threshold', 0.3),
    ('min_publish_confidence', 0.15),
    ('transform_sanity_delta_m', 0.15),
    ('transform_sanity_delta_deg', 10.0),
    # --- timeouts / fault handling ---
    ('slam_timeout_sec', 10.0),
    ('drone_timeout_sec', 300.0),
    ('confidence_decay_per_miss', 0.1),
    ('max_consecutive_miss_before_rerun', 3),
]


class MapFusionNode(Node):
    """Estimates and maintains ``T_world_slam`` and serves the aligned layer."""

    def __init__(self):
        super().__init__('map_fusion_node')

        for name, default in PARAM_SPEC:
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value
                  for name, _ in PARAM_SPEC}

        # ---- state ----
        self.drone_pre = None          # Stage 1 cache (preprocessing dict)
        self.drone_info = None         # GridInfo of the drone map
        self.pending_slam_msg = None   # last SLAM msg awaiting a drone map
        self.prior_t = None            # last accepted T_world_slam
        self.confidence = 0.0
        self.consecutive_miss = 0
        self.low_conf_streak = 0
        self.force_coarse = False
        self.aruco_world = None        # (ax, ay, ayaw) AMR pose in world
        self.last_slam_time = None
        self.last_drone_time = None
        self.slam_stale = False
        self.drone_stale = False

        # ---- QoS profiles ----
        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        best_effort = QoSProfile(
            depth=10, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)

        # ---- subscriptions ----
        self.create_subscription(OccupancyGrid, self.p['drone_map_topic'],
                                 self.drone_map_cb, latched)
        self.create_subscription(OccupancyGrid, self.p['slam_map_topic'],
                                 self.slam_map_cb, latched)
        self.create_subscription(PoseWithCovarianceStamped,
                                 self.p['aruco_pose_topic'],
                                 self.aruco_cb, best_effort)
        if self.p['use_live_tracking']:
            self.create_subscription(PoseStamped,
                                     self.p['aruco_tracking_topic'],
                                     self.aruco_tracking_cb, best_effort)

        # ---- publications ----
        self.pub_reproj = self.create_publisher(
            OccupancyGrid, self.p['reprojected_topic'], latched)
        self.pub_conf = self.create_publisher(
            Float32, self.p['confidence_topic'], best_effort)
        self.pub_status = self.create_publisher(
            String, self.p['status_topic'], best_effort)

        # ---- TF ----
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- watchdog ----
        self.create_timer(0.5, self.watchdog_cb)

        self._status('info', 'map_fusion_node started; waiting for maps.')
        self.get_logger().info('map_fusion_node ready.')

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def drone_map_cb(self, msg):
        """Stage 1: preprocess and cache the (static) drone map."""
        arr, info = occupancygrid_to_array(msg)
        self.drone_info = info
        self.drone_pre = preprocess_grid(
            arr, info,
            self.p['occupied_threshold_drone'],
            self.p['edge_kernel_size'],
            unknown_is_obstacle=False)
        self.last_drone_time = self.get_clock().now()
        self.drone_stale = False
        n_edges = len(self.drone_pre['edge_points'])
        self.get_logger().info(
            f'Drone map received: {info.width}x{info.height} @ '
            f'{info.resolution} m/cell, {n_edges} edge cells.')
        self._status('info', f'Drone map cached ({n_edges} edge cells).')

        # A SLAM map may have arrived before the drone map; process it now.
        if self.pending_slam_msg is not None:
            slam_msg = self.pending_slam_msg
            self.pending_slam_msg = None
            self.run_fusion(slam_msg)

    def slam_map_cb(self, msg):
        """Stage 2 trigger: a new SLAM map drives a full fusion cycle."""
        self.last_slam_time = self.get_clock().now()
        self.slam_stale = False
        if self.drone_pre is None:
            self.pending_slam_msg = msg
            self._status('warn',
                          'SLAM map received before drone map; deferring.')
            return
        self.run_fusion(msg)

    def aruco_cb(self, msg):
        """Store the ArUco AMR pose in the world frame (translation seed)."""
        q = msg.pose.pose.orientation
        self.aruco_world = (msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.get_logger().info(
            f'ArUco AMR pose: tx={self.aruco_world[0]:.3f} '
            f'ty={self.aruco_world[1]:.3f} yaw={self.aruco_world[2]:.3f}')

    def aruco_tracking_cb(self, msg):
        """Optional live tracking hook (Open Issue #4). Currently logs only."""
        # A full implementation would compute T_world_slam directly from the
        # drone-observed AMR pose and the AMR pose in slam_map (from TF), as a
        # high-confidence parallel estimator that bypasses ICP.
        self.get_logger().debug('Live tracking pose received (unused).')

    def watchdog_cb(self):
        """Stage 7 fault handling: detect stale inputs and warn."""
        now = self.get_clock().now()
        if self.last_slam_time is not None and not self.slam_stale:
            dt = (now - self.last_slam_time).nanoseconds * 1e-9
            if dt > self.p['slam_timeout_sec']:
                self.slam_stale = True
                self._status('warn',
                             f'SLAM map stale ({dt:.1f}s); holding last T.')
        if self.last_drone_time is not None and not self.drone_stale:
            dt = (now - self.last_drone_time).nanoseconds * 1e-9
            if dt > self.p['drone_timeout_sec']:
                self.drone_stale = True
                self._status('warn', f'Drone map stale ({dt:.1f}s).')

    # ------------------------------------------------------------------ #
    # Fusion pipeline (Stages 3 - 7)
    # ------------------------------------------------------------------ #
    def run_fusion(self, slam_msg):
        """Run Stages 2-7 for one SLAM update."""
        # --- Stage 2: preprocess the SLAM map ---
        slam_arr, slam_info = occupancygrid_to_array(slam_msg)
        slam_pre = preprocess_grid(
            slam_arr, slam_info,
            self.p['occupied_threshold_slam'],
            self.p['edge_kernel_size'],
            unknown_is_obstacle=False)
        source = slam_pre['edge_points']
        target = self.drone_pre['edge_points']

        if len(source) < 3:
            self._status('warn',
                         'SLAM map has too few edge cells; holding last T.')
            return

        # --- Stage 3: alignment decision ---
        use_coarse = (self.prior_t is None
                      or self.confidence < self.p['confidence_rerun_threshold']
                      or self.force_coarse)

        if use_coarse:
            result, ambiguous = self._run_coarse_then_icp(source, target)
            if result is None:
                self._status('warn', 'Coarse search produced no candidates.')
                return
            if ambiguous:
                self._status('warn',
                             'Ambiguous coarse alignment (possible symmetry).')
        else:
            result = icp_align(source, target, self.prior_t, self.p)

        # --- Stages 6 & 7 ---
        self._validate_and_publish(result, slam_arr, slam_info, slam_msg.header)

    def _run_coarse_then_icp(self, source, target):
        """Stage 4 then Stage 5: coarse search, refine every candidate, pick
        the one with the smallest ICP residual."""
        seed = self._compute_translation_seed()
        candidates, ambiguous = coarse_search(
            source, self.drone_pre['edge_image'], self.drone_info,
            self.p, seed=seed)
        if not candidates:
            return None, ambiguous

        best = None
        for _, cand_t in candidates:
            refined = icp_align(source, target, cand_t, self.p)
            if best is None or refined['residual'] < best['residual']:
                best = refined
        self.get_logger().info(
            f'Coarse->ICP: {len(candidates)} candidates, '
            f'best residual={best["residual"]:.4f} m.')
        return best, ambiguous

    def _compute_translation_seed(self):
        """Translation seed for the coarse search.

        The ArUco pose is the AMR in ``world``; the actual seed for
        ``T_world_slam`` is ``T_world_amr o inv(T_slam_amr)``. ``T_slam_amr``
        is looked up from TF (``slam_map`` -> ``base_link``). If that lookup
        fails we fall back to the raw ArUco position -- the search radius is
        wide enough to absorb the discrepancy. With no ArUco pose at all the
        seed is None and the coarse search spans the whole arena.
        """
        if self.aruco_world is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.p['slam_frame'], self.p['base_frame'],
                rclpy.time.Time())
            q = tf.transform.rotation
            t_slam_amr = (tf.transform.translation.x,
                          tf.transform.translation.y,
                          yaw_from_quaternion(q.x, q.y, q.z, q.w))
            seed_t = compose(self.aruco_world, invert(t_slam_amr))
            return (seed_t[0], seed_t[1])
        except TransformException:
            self.get_logger().warn(
                'slam_map->base_link TF unavailable; '
                'seeding coarse search with raw ArUco position.')
            return (self.aruco_world[0], self.aruco_world[1])

    def _validate_and_publish(self, result, slam_arr, slam_info, header):
        """Stage 6 validation + Stage 7 publish."""
        t = result['t']
        confidence = residual_to_confidence(
            result['residual'], self.p['icp_residual_scale'])

        # Stage 6.1 -- convergence check.
        if not result['converged']:
            self.consecutive_miss += 1
            self.confidence = max(
                0.0, self.confidence - self.p['confidence_decay_per_miss'])
            self._maybe_force_rerun()
            self._status('warn',
                         f'ICP did not converge (miss '
                         f'{self.consecutive_miss}); holding last T.')
            self._publish_confidence(0.0)
            return

        # Stage 6.2 -- sanity delta check against the prior.
        if self.prior_t is not None:
            d_trans = math.hypot(t[0] - self.prior_t[0],
                                 t[1] - self.prior_t[1])
            d_rot = abs(math.degrees(angle_diff(t[2], self.prior_t[2])))
            if (d_trans > self.p['transform_sanity_delta_m'] or
                    d_rot > self.p['transform_sanity_delta_deg']):
                self.force_coarse = True
                self._status('warn',
                             f'Transform jump rejected '
                             f'(d={d_trans:.3f} m, {d_rot:.1f} deg); '
                             f'holding last T.')
                self._publish_confidence(self.confidence)
                return

        # Stage 6.3 -- minimum-confidence gate.
        if confidence < self.p['min_publish_confidence']:
            self.confidence = confidence
            self._maybe_force_rerun()
            self._status('warn',
                         f'Confidence {confidence:.3f} below publish gate; '
                         f'not updating Nav2.')
            self._publish_confidence(confidence)
            return

        # --- Accepted: Stage 7 publish ---
        self.prior_t = t
        self.confidence = confidence
        self.consecutive_miss = 0
        self.low_conf_streak = 0
        self.force_coarse = False

        self._broadcast_tf(t, header.stamp)
        self._publish_reprojection(slam_arr, slam_info, t, header.stamp)
        self._publish_confidence(confidence)
        self._status('info',
                      f'T_world_slam = ({t[0]:.3f}, {t[1]:.3f}, '
                      f'{math.degrees(t[2]):.1f} deg), conf={confidence:.3f}.')

    def _maybe_force_rerun(self):
        """Force a full coarse re-search after enough misses / low-confidence
        cycles (spec Section 7)."""
        self.low_conf_streak += 1
        limit = self.p['max_consecutive_miss_before_rerun']
        if (self.consecutive_miss >= limit
                or self.low_conf_streak >= limit):
            self.force_coarse = True

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _broadcast_tf(self, t, stamp):
        """Broadcast ``T_world_slam`` as a static transform (re-sent each
        accepted cycle; the latched value is simply overwritten)."""
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.p['world_frame']
        tf.child_frame_id = self.p['slam_frame']
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(t[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def _publish_reprojection(self, slam_arr, slam_info, t, stamp):
        """Reproject the SLAM map into the world frame and publish it."""
        out_arr, out_info = reproject_slam_grid(
            slam_arr, slam_info, t, self.drone_info,
            self.p['output_resolution_m'])
        msg = array_to_occupancygrid(
            out_arr, out_info, self.p['world_frame'], stamp)
        self.pub_reproj.publish(msg)

    def _publish_confidence(self, value):
        self.pub_conf.publish(Float32(data=float(value)))

    def _status(self, level, text):
        """Publish a status string and mirror it to the node logger."""
        self.pub_status.publish(String(data=f'[{level}] {text}'))
        if level == 'warn':
            self.get_logger().warn(text)
        elif level == 'error':
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = MapFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
