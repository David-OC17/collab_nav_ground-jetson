#!/usr/bin/env python3
"""
Path Follower Node — RViz visualization only
=============================================
Subscribes to the smoothed spline path from the A* planner and animates
the robot (base_link TF) along it at a constant speed.

No cmd_vel is published — this is purely for RViz visualization.

Subscribes to:
  - /trajectory_planner/path   (nav_msgs/Path) — smoothed spline path

Publishes:
  - TF map → base_link         — robot pose animated along the path
  - /follower/robot_marker     (visualization_msgs/Marker) — robot body marker
  - /follower/progress         (std_msgs/Float32) — 0.0→1.0 path completion

Parameters:
  - linear_speed     (float, default 0.5)  m/s along the path
  - update_rate      (float, default 20.0) Hz — control + TF publish rate
  - goal_tolerance   (float, default 0.10) m  — distance to consider goal reached
  - map_frame        (str,   default 'map')
  - robot_base_frame (str,   default 'base_link')
  - path_topic       (str,   default '/trajectory_planner/path')
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from std_msgs.msg import Float32
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import tf2_ros
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped


class PathFollowerNode(Node):

    def __init__(self):
        super().__init__('path_follower')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('linear_speed',     0.5)
        self.declare_parameter('update_rate',      20.0)
        self.declare_parameter('goal_tolerance',   0.10)
        self.declare_parameter('map_frame',        'map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('path_topic',       '/trajectory_planner/path')

        self.linear_speed     = self.get_parameter('linear_speed').value
        self.update_rate      = self.get_parameter('update_rate').value
        self.goal_tolerance   = self.get_parameter('goal_tolerance').value
        self.map_frame        = self.get_parameter('map_frame').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.path_topic       = self.get_parameter('path_topic').value

        self.dt = 1.0 / self.update_rate   # seconds per update tick

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.path         = []       # list of (x, y) world coords
        self.path_index   = 0        # current target waypoint index
        self.robot_x      = 0.0
        self.robot_y      = 0.0
        self.robot_yaw    = 0.0
        self.goal_reached = True     # start idle until path arrives

        # ------------------------------------------------------------------
        # TF broadcaster
        # ------------------------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        self.path_sub = self.create_subscription(
            Path, self.path_topic, self._path_callback, reliable_qos)

        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initial_pose_callback,
            10)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.marker_pub = self.create_publisher(
            Marker, '/follower/robot_marker', reliable_qos)

        self.progress_pub = self.create_publisher(
            Float32, '/follower/progress', reliable_qos)

        
        self.reference_pub = self.create_publisher(
            Odometry, '/amr/reference', reliable_qos)

        # ------------------------------------------------------------------
        # Control timer
        # ------------------------------------------------------------------
        self.timer = self.create_timer(self.dt, self._update)

        self.get_logger().info(
            f'PathFollower ready | speed={self.linear_speed} m/s | '
            f'rate={self.update_rate} Hz | path={self.path_topic}'
        )

    # ==========================================================================
    # Path callback — new path resets follower to start
    # ==========================================================================

    def _path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn('Received path with < 2 poses — ignoring.')
            return

        self.path = [
            (p.pose.position.x,
            p.pose.position.y,
            p.pose.orientation.x,   # vx world
            p.pose.orientation.y)   # vy world
            for p in msg.poses
        ]
        
        self.path_index   = 0
        self.goal_reached = False

        # Snap robot to path start
        self.robot_x   = self.path[0][0]
        self.robot_y   = self.path[0][1]
        self.robot_yaw = self._heading_to(0)

        self.get_logger().info(
            f'New path received: {len(self.path)} waypoints — starting follower')

    
    def _initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self._broadcast_tf()   # ← immediately broadcast so planner can read it

        self.get_logger().info(
            f'Initial pose set: x={self.robot_x:.2f} y={self.robot_y:.2f} yaw={self.robot_yaw:.2f}')


    # ==========================================================================
    # Publish to cmd_vel
    # ==========================================================================    
    def _publish_cmd_vel(self):
        if self.goal_reached or not self.path or self.path_index >= len(self.path):
            self._publish_reference(0.0, 0.0, 0.0)
            return

        _, _, vx_world, vy_world = self.path[self.path_index]

        cos_y    =  math.cos(self.robot_yaw)
        sin_y    =  math.sin(self.robot_yaw)
        vx_robot =  cos_y * vx_world + sin_y * vy_world
        vy_robot = -sin_y * vx_world + cos_y * vy_world

        target_yaw = math.atan2(vy_world, vx_world)
        yaw_error  = math.atan2(
            math.sin(target_yaw - self.robot_yaw),
            math.cos(target_yaw - self.robot_yaw))
        omega = 1.5 * yaw_error

        self._publish_reference(vx_robot, vy_robot, omega)


    # Publish an Odom message type (position and velocities x,y) to the reference node for the controller
    def _publish_reference(self, vx_robot, vy_robot, omega):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.child_frame_id  = self.robot_base_frame

        msg.pose.pose.position.x    = self.robot_x
        msg.pose.pose.position.y    = self.robot_y
        msg.pose.pose.position.z    = 0.0
        msg.pose.pose.orientation.z = math.sin(self.robot_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.robot_yaw / 2.0)

        msg.twist.twist.linear.x  = vx_robot
        msg.twist.twist.linear.y  = vy_robot
        msg.twist.twist.angular.z = omega

        self.reference_pub.publish(msg)
        

    def _send_twist(self, vx, vy, omega):
        msg = Twist()
        msg.linear.x  = vx
        msg.linear.y  = vy    
        msg.angular.z = omega
        self.cmd_vel_pub.publish(msg)

    # ==========================================================================
    # Main update loop
    # ==========================================================================
    def _update(self):
        self._broadcast_tf()
        self._publish_robot_marker()

        if self.goal_reached or not self.path:
            self._publish_reference(0.0, 0.0, 0.0)   # ← was _send_twist
            return

        _, _, vx_world, vy_world = self.path[self.path_index]
        current_speed  = math.hypot(vx_world, vy_world)
        current_speed  = max(current_speed, 0.05)
        step_remaining = current_speed * self.dt

        while step_remaining > 0 and self.path_index < len(self.path):
            tx, ty = self.path[self.path_index][0], self.path[self.path_index][1]
            dx   = tx - self.robot_x
            dy   = ty - self.robot_y
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                self.path_index += 1
                continue

            if dist <= step_remaining:
                step_remaining -= dist
                self.robot_x    = tx
                self.robot_y    = ty
                self.path_index += 1
            else:
                ratio           = step_remaining / dist
                self.robot_x   += ratio * dx
                self.robot_y   += ratio * dy
                step_remaining  = 0

            if self.path_index < len(self.path):
                nx = self.path[self.path_index][0]
                ny = self.path[self.path_index][1]
                self.robot_yaw = math.atan2(ny - self.robot_y, nx - self.robot_x)

        if self.path_index >= len(self.path):
            gx, gy = self.path[-1][0], self.path[-1][1]
            if math.hypot(self.robot_x - gx, self.robot_y - gy) < self.goal_tolerance:
                self.robot_x      = gx
                self.robot_y      = gy
                self.goal_reached = True
                self._publish_reference(0.0, 0.0, 0.0)   # ← was _send_twist
                self.get_logger().info('Goal reached.')
                self._publish_progress(1.0)
                return

        self._publish_cmd_vel()

        progress = self.path_index / max(len(self.path) - 1, 1)
        self._publish_progress(progress)

    # ==========================================================================
    # Heading helper — face toward the next waypoint
    # ==========================================================================

    def _heading_to(self, index: int) -> float:
        if index >= len(self.path):
            return self.robot_yaw
        tx = self.path[index][0]   # safe for 4-tuples
        ty = self.path[index][1]
        return math.atan2(ty - self.robot_y, tx - self.robot_x)

    # ==========================================================================
    # TF broadcast
    # ==========================================================================

    def _broadcast_tf(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.map_frame
        tf_msg.child_frame_id  = self.robot_base_frame

        tf_msg.transform.translation.x = self.robot_x
        tf_msg.transform.translation.y = self.robot_y
        tf_msg.transform.translation.z = 0.0

        # Yaw → quaternion (rotation around Z only)
        tf_msg.transform.rotation.z = math.sin(self.robot_yaw / 2.0)
        tf_msg.transform.rotation.w = math.cos(self.robot_yaw / 2.0)

        self.tf_broadcaster.sendTransform(tf_msg)

    # ==========================================================================
    # Robot body marker — arrow showing position + heading
    # ==========================================================================

    def _publish_robot_marker(self):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns     = 'follower'
        m.id     = 0
        m.type   = Marker.ARROW
        m.action = Marker.ADD

        m.pose.position.x    = self.robot_x
        m.pose.position.y    = self.robot_y
        m.pose.position.z    = 0.05
        m.pose.orientation.z = math.sin(self.robot_yaw / 2.0)
        m.pose.orientation.w = math.cos(self.robot_yaw / 2.0)

        m.scale.x = 0.35   # arrow length
        m.scale.y = 0.12   # arrow width
        m.scale.z = 0.10   # arrow height

        # Colour: green while moving, grey when idle
        if self.goal_reached or not self.path:
            m.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8)  # grey — idle
        else:
            progress = self.path_index / max(len(self.path) - 1, 1)
            m.color  = ColorRGBA(
                r=progress,
                g=1.0 - progress * 0.5,
                b=0.2,
                a=1.0
            )  # green → orange as it approaches goal

        self.marker_pub.publish(m)

    # ==========================================================================
    # Progress publisher
    # ==========================================================================

    def _publish_progress(self, value: float):
        msg = Float32()
        msg.data = float(value)
        self.progress_pub.publish(msg)


# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()