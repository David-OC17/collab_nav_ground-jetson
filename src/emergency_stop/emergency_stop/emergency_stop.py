import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import math 

class EmergencyStopNode(Node):
    def __init__(self):
        super().__init__('emergency_stop')

        # Parameters
        self.declare_parameter('min_obstacle_distance_m', 0.4)
        self.declare_parameter('max_linear_speed_mps', 1.5)
        # Blind spot (robot chassis is detected in this range)
        self.declare_parameter('mask_angles_deg', [120.0, 240.0])  
        # Filter noise
        self.declare_parameter('trigger_count',           5)    # scans in a row before triggering
        self.declare_parameter('clear_count',             5)    # scans in a row before

        self.min_dist = self.get_parameter('min_obstacle_distance_m').value
        self.max_speed = self.get_parameter('max_linear_speed_mps').value
        self._trigger_n = self.get_parameter('trigger_count').value
        self._clear_n   = self.get_parameter('clear_count').value
        mask = self.get_parameter('mask_angles_deg').value

        self._mask_min = math.radians(mask[0])
        self._mask_max = math.radians(mask[1])

        # Confirmation counters
        self._obstacle_counter = 0
        self._clear_counter    = 0

        # State
        self._stop_active = False
        self._stop_reasons = set()

        # Subscriptions — add more triggers here as needed
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        # self.create_subscription(Odometry, '/amr/ekf/odom', self._odom_cb, 10)

        # Publishers
        self._pub_stop   = self.create_publisher(Bool, '/amr/emergency_stop', 10)

        # Publish state at 10 Hz regardless of trigger changes
        self.create_timer(0.1, self._publish_state)
        self.get_logger().info('EmergencyStopNode ready')

    # ── Trigger: LiDAR too close ─────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        valid = []
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if self._mask_min <= angle <= self._mask_max:   # ← self. prefix
                continue
            if msg.range_min < r < msg.range_max:
                valid.append(r)

        obstacle_detected = valid and min(valid) < self.min_dist

        if obstacle_detected:
            self._obstacle_counter += 1
            self._clear_counter = 0
            if self._obstacle_counter >= self._trigger_n:
                self._set_stop(True, 'lidar_proximity')
        else:
            self._clear_counter += 1
            self._obstacle_counter = 0
            if self._clear_counter >= self._clear_n:
                self._clear_stop('lidar_proximity')


    # ── Trigger: abnormal speed (wheel slip / runaway) ───────────────────
    def _odom_cb(self, msg: Odometry):
        speed = abs(msg.twist.twist.linear.x)
        if speed > self.max_speed:
            self._set_stop(True, 'speed_limit')
        else:
            self._clear_stop('speed_limit')

    # ── State management ─────────────────────────────────────────────────
    def _set_stop(self, active: bool, reason: str):
        if active:
            self._stop_reasons.add(reason)
        else:
            self._stop_reasons.discard(reason)
        was_active = self._stop_active
        self._stop_active = bool(self._stop_reasons)
        if self._stop_active != was_active:
            state = 'ACTIVE' if self._stop_active else 'CLEARED'
            self.get_logger().warn(
                f'Emergency stop {state} — reasons: {self._stop_reasons}')

    def _clear_stop(self, reason: str):
        self._set_stop(False, reason)

    def _publish_state(self):
        msg = Bool(data=self._stop_active)
        self._pub_stop.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()