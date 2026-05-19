"""``mock_arena`` -- a self-contained test harness for :mod:`map_fusion_node`.

It fabricates a 4x4 m arena and publishes, with no hardware required:

* ``/drone/map``       -- a full-coverage drone OccupancyGrid (filled blobs),
                          in the ``world`` frame, published once (latched).
* ``/map``             -- a SLAM Toolbox-style OccupancyGrid (contours only),
                          in the ``slam_map`` frame, revealed progressively at
                          0.2 Hz to imitate an exploring AMR.
* ``/aruco/amr_pose``  -- the AMR pose in ``world``, published once.
* TF ``slam_map``->``base_link`` -- the AMR pose inside the SLAM frame, so the
                          node's translation-seed lookup succeeds. (On a real
                          robot this edge comes from SLAM Toolbox + the EKF.)

The SLAM frame is offset from the world frame by a configurable *ground-truth*
transform; the node should recover it. The ground truth is printed at startup
so the estimate can be checked.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from tf2_ros import TransformBroadcaster

from .geometry import apply_se2, invert, quaternion_from_yaw
from .grid_utils import GridInfo, array_to_occupancygrid

# Arena obstacles in the world frame, as (kind, params...) metric tuples.
_OBSTACLES = [
    ('rect', 1.00, 1.00, 1.50, 1.60),   # box A
    ('rect', 2.40, 2.00, 3.10, 2.50),   # box B
    ('circle', 2.00, 3.10, 0.18),       # cone
]
_ARENA = 4.0
_WALL_T = 0.12
_AMR_WORLD = (0.70, 0.60, 0.50)         # AMR pose (x, y, yaw) in world


def _build_truth(res):
    """Boolean obstacle occupancy of the arena at resolution ``res``."""
    n = int(round(_ARENA / res))
    xs = (np.arange(n) + 0.5) * res
    ys = (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    occ = np.zeros((n, n), dtype=bool)
    # Border walls.
    occ |= (gx < _WALL_T) | (gx > _ARENA - _WALL_T)
    occ |= (gy < _WALL_T) | (gy > _ARENA - _WALL_T)
    # Interior obstacles.
    for ob in _OBSTACLES:
        if ob[0] == 'rect':
            _, x0, y0, x1, y1 = ob
            occ |= (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        else:
            _, cx, cy, r = ob
            occ |= ((gx - cx) ** 2 + (gy - cy) ** 2) <= r * r
    return occ


def _morph_gradient(mask_uint8):
    """Contour of a binary mask: ``mask XOR erode(mask)`` (3x3 kernel)."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.bitwise_xor(mask_uint8, cv2.erode(mask_uint8, kernel))


class MockArena(Node):
    """Publishes a synthetic arena for testing the fusion node."""

    def __init__(self):
        super().__init__('mock_arena')

        self.declare_parameter('gt_tx', 1.30)
        self.declare_parameter('gt_ty', -0.80)
        self.declare_parameter('gt_theta_deg', 35.0)
        self.declare_parameter('drone_res', 0.02)
        self.declare_parameter('slam_res', 0.05)
        self.declare_parameter('slam_period_sec', 5.0)
        self.declare_parameter('aruco_delay_sec', 6.0)
        self.declare_parameter('reveal_start_m', 1.0)
        self.declare_parameter('reveal_growth_m', 0.6)

        gp = self.get_parameter
        # Ground-truth T_world_slam: maps slam_map -> world.
        self.gt = (gp('gt_tx').value, gp('gt_ty').value,
                   math.radians(gp('gt_theta_deg').value))
        self.drone_res = gp('drone_res').value
        self.slam_res = gp('slam_res').value
        self.reveal_start = gp('reveal_start_m').value
        self.reveal_growth = gp('reveal_growth_m').value

        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_drone = self.create_publisher(OccupancyGrid, '/drone/map',
                                               latched)
        self.pub_slam = self.create_publisher(OccupancyGrid, '/map', latched)
        self.pub_aruco = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/amr_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Pre-compute the static drone map and the SLAM-frame obstacle field.
        self.truth = _build_truth(self.drone_res)
        self._build_slam_field()
        self.reveal_tick = 0

        self._publish_drone_map()
        self.create_timer(gp('slam_period_sec').value, self._publish_slam_map)
        self.create_timer(0.1, self._publish_tf)
        self._aruco_timer = self.create_timer(
            gp('aruco_delay_sec').value, self._publish_aruco_once)

        g = self.gt
        self.get_logger().info(
            '=== mock_arena ground truth ===\n'
            f'  T_world_slam = (tx={g[0]:.3f}, ty={g[1]:.3f}, '
            f'theta={math.degrees(g[2]):.1f} deg)\n'
            '  The fusion node should converge close to these values.')

    # ------------------------------------------------------------------ #
    def _build_slam_field(self):
        """Build the SLAM-frame obstacle contour grid and its geometry."""
        corners_w = np.array([[0.0, 0.0], [_ARENA, 0.0],
                              [_ARENA, _ARENA], [0.0, _ARENA]])
        corners_s = apply_se2(invert(self.gt), corners_w)
        pad = 0.20
        smin = corners_s.min(axis=0) - pad
        smax = corners_s.max(axis=0) + pad
        w = int(math.ceil((smax[0] - smin[0]) / self.slam_res))
        h = int(math.ceil((smax[1] - smin[1]) / self.slam_res))
        self.slam_info = GridInfo(self.slam_res, w, h,
                                  float(smin[0]), float(smin[1]), 0.0)

        # Sample the world truth at every SLAM cell centre.
        ii, jj = np.meshgrid(np.arange(w), np.arange(h))
        cx = smin[0] + (ii + 0.5) * self.slam_res
        cy = smin[1] + (jj + 0.5) * self.slam_res
        centres_s = np.stack([cx.ravel(), cy.ravel()], axis=-1)
        world = apply_se2(self.gt, centres_s)
        ti = np.clip((world[:, 0] / self.drone_res).astype(int),
                     0, self.truth.shape[1] - 1)
        tj = np.clip((world[:, 1] / self.drone_res).astype(int),
                     0, self.truth.shape[0] - 1)
        in_arena = ((world[:, 0] >= 0) & (world[:, 0] < _ARENA) &
                    (world[:, 1] >= 0) & (world[:, 1] < _ARENA))
        obstacle = (self.truth[tj, ti] & in_arena).reshape(h, w)

        self.slam_contour = _morph_gradient(obstacle.astype(np.uint8))
        self.slam_in_arena = in_arena.reshape(h, w)

        # AMR position inside the SLAM frame (for the reveal disk + TF).
        self.amr_slam = apply_se2(invert(self.gt),
                                  np.array([_AMR_WORLD[:2]]))[0]
        self.amr_slam_yaw = _AMR_WORLD[2] - self.gt[2]
        self._slam_centres = np.stack([cx, cy], axis=-1)

    # ------------------------------------------------------------------ #
    def _publish_drone_map(self):
        n = self.truth.shape[0]
        arr = np.where(self.truth, np.int8(100), np.int8(0))
        info = GridInfo(self.drone_res, n, n, 0.0, 0.0, 0.0)
        msg = array_to_occupancygrid(arr, info, 'world',
                                     self.get_clock().now().to_msg())
        self.pub_drone.publish(msg)
        self.get_logger().info(f'Published drone map ({n}x{n}).')

    def _publish_slam_map(self):
        """Publish a progressively revealed SLAM contour map."""
        radius = self.reveal_start + self.reveal_tick * self.reveal_growth
        self.reveal_tick += 1
        d = np.linalg.norm(self._slam_centres - self.amr_slam, axis=-1)
        revealed = (d <= radius) & self.slam_in_arena

        arr = np.full(self.slam_contour.shape, np.int8(-1))
        arr[revealed] = 0
        arr[revealed & (self.slam_contour == 1)] = 100

        msg = array_to_occupancygrid(arr, self.slam_info, 'slam_map',
                                     self.get_clock().now().to_msg())
        self.pub_slam.publish(msg)
        known = int((arr != -1).sum())
        self.get_logger().info(
            f'Published SLAM map (reveal r={radius:.2f} m, {known} known cells).')

    def _publish_tf(self):
        """Broadcast slam_map -> base_link (the AMR pose in the SLAM frame)."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'slam_map'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = float(self.amr_slam[0])
        tf.transform.translation.y = float(self.amr_slam[1])
        qx, qy, qz, qw = quaternion_from_yaw(self.amr_slam_yaw)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def _publish_aruco_once(self):
        """Publish the AMR's world-frame pose, then cancel the timer."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.pose.position.x = _AMR_WORLD[0]
        msg.pose.pose.position.y = _AMR_WORLD[1]
        qx, qy, qz, qw = quaternion_from_yaw(_AMR_WORLD[2])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Small diagonal covariance (high-confidence averaged detection).
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.0025
        cov[35] = 0.0025
        msg.pose.covariance = cov
        self.pub_aruco.publish(msg)
        self.get_logger().info(
            f'Published ArUco AMR pose ({_AMR_WORLD[0]:.2f}, '
            f'{_AMR_WORLD[1]:.2f}) in world.')
        self._aruco_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = MockArena()
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
