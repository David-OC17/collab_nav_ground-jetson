"""
mission_orchestrator.orchestrator_node
────────────────────────────────────────────────────────────────────────────
End-to-end mission sequencer for the collab_nav ground-robot + drone system.

Major stages and substages
───────────────────────────
 01  Optitrack bringup
     01.a  Check /optitrack/rigid_body presence + header; launch client if absent
     01.b  Sanity-check /optitrack/rigid_body contents (frame_id + freshness)
 02  Arena map builder bringup
     02.a  Launch server + configure background_path parameter
     02.b  Configure online/offline mode (default online)
     (Async) 02.c  Server responds with OccupancyGrid + marker poses (joined at 04.b)
 03  Drone routine
     03.a  Connect to Tello WiFi (nmcli scan + connect)
     03.b  Launch tello_driver
     03.c  Drone preflight: verify /camera/image_raw live, /battery_state ≥ min %
     03.d  Launch tello_map (drone takes off + scanning routine)
     03.e  (If online) Send online_start to map builder (consume /camera/image_proc)
     03.f  Observe drone state transitions 1→2→3→4 with per-stage timeouts
     03.g  Wait for scan.mp4 + telemetry.csv (fresh within max_age_sec)
     03.h  Verify scan.mp4 integrity via ffmpeg
 04  Drone Aruco localizer
     (kickoff) online_stop (if online) + send BuildArenaMap goal in the BACKGROUND
     04    Launch arena_marker_localizer service node + wait for readiness
     04.a  Call /localize_markers (ORIENTATION source) — runs while the map builds
     04.b  Join the map-builder result (POSITION source) and publish
           /aruco/amr/pose (POSITION from the map-builder, ORIENTATION from the
           localizer). If the AMR position/orientation is unavailable, publish an
           all-NaN pose -- alignment_node reads that as "no data" and falls back
           to the trivial identity transform. The OccupancyGrid and goal pose are
           published by 04.c, gated on the classifier.
    04.c   Run arena_map_builder's map classifier (pass/fail) on the BuildArenaMap
           diagnostic features. PASS -> publish the OccupancyGrid to /drone/map
           and /aruco/goal/pose, continue the regular pipeline. FAIL (map
           unavailable or classifier reject) -> publish a border template grid to
           /drone/map (no goal pose), the planner uses /fusion/map (stage 11), and
           the frontier-exploration stack is launched at 11.b and triggered at 12.a.
 05  Isaac ROS Visual SLAM (cuSLAM) bringup
     05.a  Verify Intel RealSense D435i is plugged in
     05.b  Start Docker container + launch visual SLAM (via start_vslam.sh)
     05.c  Check /visual_slam/tracking/odometry has a valid output
 06  Rasp bringup
     06.a  Ping Raspberry Pi
     06.b  SSH connect to Raspberry Pi
     06.c  Start amr_bringup systemd service; verify active
     06.d  Wait for /imu/data_raw to publish N messages
 07  Emergency stop bringup
     07.a  Launch emergency_stop node; verify /amr/emergency_stop is inactive
 08  AMR Aruco localizer
     08.a  Launch aruco_localizer (/aruco_pose → EKF). Camera source follows
           vslam.enabled: RealSense color when disabled, the VSLAM RealSense IR
           stream when enabled.
 09  Mapping bringup
     09.a  Launch oradar lidar
     09.b  Launch alignment_node (/aruco/amr/pose → world->odom tf)
     09.c  Launch odom-based mapper (no SLAM)
 10  Map fusion
     Launch fusion.launch.py (overlays the drone map and the AMR-built map);
     fire-and-forget — no readiness wait (fusion only emits once /drone/map
     exists, so there is nothing to block on).
 11  Trajectory planner bringup (astar_planner2 + spline_follower)
     map_topic = /drone/map on PASS, /fusion/map on FAIL (the dumped-map case)
     11.b (FAIL only) Launch the frontier-exploration fallback stack; it idles
          until the 12.a trigger. astar_planner2 + spline_follower come from 11,
          not from this stack (single planner instance).
 12  Enter observer mode and log updates
     12.a If we are in frontier exploration mode (map FAILED), send True on
          /map_fail_fallback/start once the rest of the bringup is complete

From stage 12 onward the orchestrator only observes; the mapper and
trajectory_planner operate autonomously.

Map-builder mode (config map_builder.online, default online)
────────────────────────────────────────────────────────────
 online (default): the server is launched in online mode at stage 02 and its
   online_start service is called at 03.e; it stitches incrementally from the
   drone's processed-image stream during the flight. After the drone lands,
   online_stop finalizes the stitch and the BuildArenaMap goal (transfer +
   occupancy only) is sent in the background; 04.b joins it.
 offline: the server is launched in plain mode at stage 02; after the drone
   lands the BuildArenaMap goal (full stitch + transfer + occupancy) is sent in
   the background and joined at 04.b. The processed-image stream is ignored.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

import yaml
import paramiko

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
    HistoryPolicy,
)

from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import BatteryState, Image, Imu
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger

from arena_map_builder_msgs.action import BuildArenaMap
from arena_marker_localizer_interfaces.srv import LocalizeMarkers

from ros2_security import SecureNodeMixin


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class MissionAbortError(RuntimeError):
    """Raised to unwind the orchestration stack and trigger the abort sequence."""


# ─────────────────────────────────────────────────────────────────────────────
# Process helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kill_proc(
    proc: subprocess.Popen,
    name: str,
    log: logging.Logger,
    sigint_timeout: float = 5.0,
) -> None:
    """Send SIGINT; if the process does not exit within *sigint_timeout* seconds
    send SIGKILL.  Safe to call on an already-dead process."""
    if proc is None or proc.poll() is not None:
        return
    log.info(f"Sending SIGINT to '{name}' (pid={proc.pid})")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=sigint_timeout)
        log.info(f"Process '{name}' exited cleanly after SIGINT.")
    except subprocess.TimeoutExpired:
        log.warning(f"'{name}' still alive after {sigint_timeout}s — sending SIGKILL")
        proc.kill()
        proc.wait()
        log.info(f"Process '{name}' killed (SIGKILL).")


# ─────────────────────────────────────────────────────────────────────────────
# Observer: message formatters + topic registry
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _fmt_odometry(msg: Odometry) -> str:
    pos = msg.pose.pose.position
    yaw = _yaw_from_quat(msg.pose.pose.orientation)
    v   = msg.twist.twist.linear.x
    w   = msg.twist.twist.angular.z
    return (f"x={pos.x:7.3f}  y={pos.y:7.3f}  θ={math.degrees(yaw):7.2f}°"
            f"  v={v:6.3f}  ω={w:6.3f}")


def _fmt_pose_with_cov(msg: PoseWithCovarianceStamped) -> str:
    pos = msg.pose.pose.position
    yaw = _yaw_from_quat(msg.pose.pose.orientation)
    return f"x={pos.x:7.3f}  y={pos.y:7.3f}  θ={math.degrees(yaw):7.2f}°"


def _fmt_twist(msg: Twist) -> str:
    return f"v={msg.linear.x:6.3f}  ω={msg.angular.z:6.3f}"


def _fmt_imu(msg) -> str:  # sensor_msgs/Imu
    a = msg.linear_acceleration
    w = msg.angular_velocity
    return (f"ax={a.x:6.3f}  ay={a.y:6.3f}  az={a.z:6.3f}"
            f"  ωz={w.z:6.3f}")


def _fmt_point(msg: Point) -> str:
    return f"x={msg.x:7.3f}  y={msg.y:7.3f}  z={msg.z:7.3f}"


# Registry entry: (qos_key, display_label, msg_type, formatter)
# qos_key: 'latched' for TRANSIENT_LOCAL topics, 'default' for everything else.
_OBSERVER_REGISTRY: Dict[str, tuple] = {
    '/amr/reference':   ('default', 'ref',       Odometry,                _fmt_odometry),
    '/amr/ekf/odom':    ('default', 'ekf/odom',  Odometry,                _fmt_odometry),
    '/visual_slam/tracking/odometry': ('default', 'vslam', Odometry,      _fmt_odometry),
    '/aruco/amr/pose':  ('latched', 'amr_pose',  PoseWithCovarianceStamped, _fmt_pose_with_cov),
    '/aruco/goal/pose': ('latched', 'goal_pose', PoseWithCovarianceStamped, _fmt_pose_with_cov),
    '/amr/vel_raw':     ('default', 'vel_raw',   Twist,                   _fmt_twist),
    '/imu/data_raw':    ('default', 'imu',       Imu,                     _fmt_imu),
    '/amr/error':       ('default', 'error',     Point,                   _fmt_point),
}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator node
# ─────────────────────────────────────────────────────────────────────────────

class MissionOrchestratorNode(SecureNodeMixin, Node):
    """ROS 2 node that sequentially executes all mission stages."""

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, config_path: str) -> None:
        super().__init__('mission_orchestrator')
        self.declare_parameter('certs_dir', './certs')
        self.security_init(certs_dir=self.get_parameter('certs_dir').value)

        self._cfg: dict = self._load_config(config_path)
        self._log: logging.Logger = self._setup_logging()

        # Subprocess handles
        self._processes: Dict[str, subprocess.Popen] = {}
        self._rosbag_proc: Optional[subprocess.Popen] = None

        # SSH state
        self._ssh: Optional[paramiko.SSHClient] = None

        # Mission state flags
        self._mission_complete = False
        self._drone_aborted = False

        # Map-builder mode: online (stitch live during the flight, default) vs
        # offline (stitch from the saved scan.mp4 afterwards).
        self._online_enabled = bool(
            self._cfg.get('map_builder', {}).get('online', True))

        # Background BuildArenaMap state (sent after the drone lands, joined at
        # stage 04.b so the transfer/occupancy — and the full stitch in offline
        # — overlaps the marker-localizer call).
        self._map_goal_handle = None
        self._map_result_future = None
        self._map_send_error: Optional[str] = None

        # Diagnostic feature vector returned by BuildArenaMap (the full 46-feature
        # map-quality set, same keys as the offline sweep's metrics.yaml).
        # Populated in _await_map_result; consumed by the runtime classifier.
        self._map_features: Dict[str, float] = {}

        # Map-quality decision (stage 04.c). When the classifier rejects the
        # stitched map, _map_failed is True: the OccupancyGrid + goal pose are not
        # published, the planner uses the fusion map, and frontier exploration is
        # launched after stage 09 and triggered at 10.a.
        self._classifier = None          # lazily constructed MapQualityClassifier
        self._map_failed: bool = False

        # ── ROS 2 sync primitives ──
        self._imu_ready_event = threading.Event()
        self._imu_msg_count: int = 0

        self._optitrack_event = threading.Event()
        self._optitrack_last_msg: Optional[PoseStamped] = None
        self._optitrack_lock = threading.Lock()

        self._drone_state: Optional[int] = None
        self._drone_state_cond = threading.Condition(threading.Lock())

        self._camera_event = threading.Event()
        self._battery_event = threading.Event()
        self._battery_pct: Optional[float] = None

        self._video_path: Optional[str] = None
        self._telemetry_path: Optional[str] = None

        self._estop_event = threading.Event()
        self._estop_lock = threading.Lock()
        self._estop_count: int = 0
        self._estop_triggered: bool = False

        self._drone_map_event = threading.Event()
        self._vslam_odom_event = threading.Event()

        # Observer state (populated by _start_observer after mission handoff)
        self._observer_cache: Dict[str, object] = {}
        self._observer_lock = threading.Lock()
        self._observer_topics: List[str] = []

        self._init_ros_interfaces()
        self._log.info("MissionOrchestratorNode initialised.")

    # ── Configuration ────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as fh:
            raw = yaml.safe_load(fh)
        cfg = raw.get('orchestrator', raw)
        self.get_logger().info(f"Config loaded from {path}")
        return cfg

    def _setup_logging(self) -> logging.Logger:
        cfg_log = self._cfg.get('logging', {})
        log_dir = cfg_log.get('log_dir', '/tmp/mission_orchestrator_logs')
        level = getattr(logging, cfg_log.get('log_level', 'INFO').upper(), logging.INFO)

        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'mission_{ts}.log')

        log = logging.getLogger('mission_orchestrator')
        log.setLevel(level)
        log.handlers.clear()

        fmt = logging.Formatter('[%(asctime)s][%(levelname)-7s] %(message)s',
                                datefmt='%H:%M:%S')
        for handler in (logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            log.addHandler(handler)

        log.info(f"Log file: {log_file}")
        return log

    # ── ROS 2 interface setup ────────────────────────────────────────────────

    def _init_ros_interfaces(self) -> None:
        cfg = self._cfg
        best_effort = QoSProfile(
            depth=10, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Native subscriptions: uncontrolled hardware / third-party nodes
        self.create_subscription(Imu,
            cfg['imu']['topic'], self._imu_cb, 10)
        self.create_subscription(Int32,
            cfg['drone']['state_topic'], self._drone_state_cb, 10)
        self.create_subscription(Image,
            cfg['drone']['camera_topic'], self._camera_cb, best_effort)
        self.create_subscription(BatteryState,
            cfg['drone']['battery_topic'], self._battery_cb, 10)
        self.create_subscription(Odometry,
            cfg.get('vslam', {}).get('odometry_topic', '/visual_slam/tracking/odometry'),
            self._vslam_odom_cb, best_effort)
        # Secure subscriptions: messages from controlled nodes
        self.create_secure_subscription(
            cfg['optitrack']['topic'], PoseStamped, self._optitrack_cb, min_level=None, qos=qos)
        self.create_secure_subscription(
            cfg.get('emergency_stop', {}).get('topic', '/amr/emergency_stop'),
            Bool, self._emergency_stop_cb, min_level=None, qos=10)
        self.create_secure_subscription(
            cfg['map_builder']['drone_map_topic'], OccupancyGrid, self._drone_map_cb, min_level=None, qos=latched)

        self._pub_aruco_amr = self.create_secure_publisher(cfg['aruco']['amr_pose_topic'], PoseWithCovarianceStamped, latched)
        self._pub_aruco_goal = self.create_secure_publisher(cfg['aruco']['goal_pose_topic'], PoseWithCovarianceStamped, latched)
        self._pub_drone_map = self.create_secure_publisher(cfg['map_builder']['drone_map_topic'], OccupancyGrid, latched)

        # Frontier-exploration fallback trigger (stage 10.a). Reliable + VOLATILE
        # to match explorer_controller's /map_fail_fallback/start subscription
        # (it deliberately does not replay to late joiners), so we publish only
        # after confirming the subscriber is up.
        fallback_qos = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._pub_map_fail_start = self.create_secure_publisher(
            cfg.get('frontier', {}).get('start_topic', '/map_fail_fallback/start'),
            Bool, fallback_qos)

        self._loc_client = self.create_client(
            LocalizeMarkers, cfg['marker_localizer']['service_name'])
        self._map_action_client = ActionClient(
            self, BuildArenaMap, cfg['map_builder']['action_name'])

        # Online-stitching control services (only used when map_builder.online).
        mb = cfg['map_builder']
        self._online_start_cli = self.create_client(
            Trigger, mb.get('online_start_service',
                            '/build_arena_map_server/online_start'))
        self._online_stop_cli = self.create_client(
            Trigger, mb.get('online_stop_service',
                            '/build_arena_map_server/online_stop'))

    # ────────────────────────────────────────────────────────────────────────
    # ROS 2 callbacks
    # ────────────────────────────────────────────────────────────────────────

    def _imu_cb(self, _msg: Imu) -> None:
        self._imu_msg_count += 1
        if (self._imu_msg_count >= self._cfg['imu']['message_count']
                and not self._imu_ready_event.is_set()):
            self._imu_ready_event.set()

    def _optitrack_cb(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != self._cfg['optitrack']['expected_frame_id']:
            return
        with self._optitrack_lock:
            self._optitrack_last_msg = msg
        self._optitrack_event.set()

    def _drone_state_cb(self, msg: Int32) -> None:
        state = int(msg.data)
        with self._drone_state_cond:
            self._drone_state = state
            self._drone_state_cond.notify_all()
        self._log.debug(f"Drone state → {state}")

    def _camera_cb(self, _msg: Image) -> None:
        self._camera_event.set()

    def _battery_cb(self, msg: BatteryState) -> None:
        self._battery_pct = float(msg.percentage)
        self._battery_event.set()

    def _emergency_stop_cb(self, msg: Bool) -> None:
        if self._estop_event.is_set():
            return
        with self._estop_lock:
            if self._estop_event.is_set():
                return
            if msg.data:
                self._estop_triggered = True
            self._estop_count += 1
            n = self._cfg.get('emergency_stop', {}).get('check_count', 10)
            if self._estop_triggered or self._estop_count >= n:
                self._estop_event.set()

    def _drone_map_cb(self, _msg: OccupancyGrid) -> None:
        self._drone_map_event.set()

    def _vslam_odom_cb(self, _msg: Odometry) -> None:
        self._vslam_odom_event.set()

    # ────────────────────────────────────────────────────────────────────────
    # Service / action helpers (called from main thread; executor spins in bg)
    # ────────────────────────────────────────────────────────────────────────

    def _call_service(self, client, request, timeout_sec: float):
        """Call a ROS 2 service synchronously from the main thread."""
        deadline = time.monotonic() + timeout_sec
        while not client.wait_for_service(timeout_sec=1.0):
            if time.monotonic() > deadline:
                raise MissionAbortError(
                    f"Service '{client.srv_name}' not available after {timeout_sec}s")
            self._log.info(f"  Waiting for service {client.srv_name} …")

        future = client.call_async(request)
        done_evt = threading.Event()
        future.add_done_callback(lambda _: done_evt.set())
        remaining = max(1.0, deadline - time.monotonic())
        if not done_evt.wait(timeout=remaining):
            raise MissionAbortError(
                f"Service call to '{client.srv_name}' timed out after {timeout_sec}s")
        return future.result()

    def _map_action_feedback(self, fb_msg) -> None:
        fb = fb_msg.feedback
        self._log.info(f"  [map_builder] [{fb.stage}] {fb.progress * 100:.0f}%  {fb.message}")

    # ────────────────────────────────────────────────────────────────────────
    # Abort helpers
    # ────────────────────────────────────────────────────────────────────────

    def _abort_drone(self) -> None:
        """Safe drone land sequence: stop controller → /land → wait → kill driver."""
        if self._drone_aborted:
            return
        self._drone_aborted = True
        self._log.warning("══ DRONE ABORT SEQUENCE ══")

        self._log.info("  Killing tello_map (tello_controller) …")
        _kill_proc(self._processes.get('tello_map'), 'tello_map', self._log)

        # Publish /land via subprocess — avoids dependency on ROS context validity
        self._log.info("  Sending /land command …")
        try:
            subprocess.run(
                ['ros2', 'topic', 'pub', '-1', '-w', '1',
                 '/land', 'std_msgs/msg/Empty', '{}'],
                timeout=10.0,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._log.info("  /land sent.")
        except subprocess.TimeoutExpired:
            self._log.warning("  /land publish timed out (tello_driver may already be down)")

        land_wait = float(self._cfg['drone'].get('land_wait_sec', 30.0))
        self._log.info(f"  Keeping tello_driver alive for {land_wait}s while drone lands …")
        time.sleep(land_wait)

        self._log.info("  Killing tello_driver …")
        _kill_proc(self._processes.get('tello_driver'), 'tello_driver', self._log)

        self._log.warning("══ DRONE ABORT COMPLETE ══")

    def _teardown_ssh(self) -> None:
        """Stop the AMR service and close the SSH connection. Idempotent."""
        if self._ssh is None:
            return
        try:
            svc = self._cfg['rasp']['amr_service']
            self._ssh.exec_command(f'systemctl stop {svc}')
            self._log.info(f"  Stopped {svc} on Raspberry Pi")
        except Exception:
            pass
        try:
            self._ssh.close()
        except Exception:
            pass
        self._ssh = None

    def _start_rosbag(self) -> None:
        cfg_rb = self._cfg.get('rosbag', {})
        if not cfg_rb.get('enabled', False):
            return

        output_dir = cfg_rb.get('output_dir', '/tmp/mission_orchestrator_logs')
        topics: List[str] = cfg_rb.get('topics') or []

        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        bag_path = os.path.join(output_dir, f'mission_{ts}')

        cmd = ['ros2', 'bag', 'record', '-o', bag_path]
        if topics:
            cmd.extend(topics)
            self._log.info(f"  Recording {len(topics)} topic(s): {topics}")
        else:
            cmd.append('-a')
            self._log.info("  Recording all topics (-a)")

        self._rosbag_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._log.info(f"Rosbag recording started → {bag_path}  (pid={self._rosbag_proc.pid})")

    def _stop_rosbag(self) -> None:
        if self._rosbag_proc is None:
            return
        _kill_proc(self._rosbag_proc, 'rosbag', self._log)
        self._rosbag_proc = None

    def _abort(self) -> None:
        """Kill all processes and close SSH.  Idempotent."""
        self._log.error("══════ MISSION ABORT ══════")
        self._abort_drone()
        self._stop_rosbag()
        for name, proc in list(self._processes.items()):
            _kill_proc(proc, name, self._log)
        self._teardown_ssh()
        self._log.error("══════ ABORT COMPLETE ══════")

    # ────────────────────────────────────────────────────────────────────────
    # Shared low-level helpers
    # ────────────────────────────────────────────────────────────────────────

    def _ssh_run(self, cmd: str) -> tuple[int, str, str]:
        """Run a command over SSH; return (exit_code, stdout, stderr)."""
        _in, out, err = self._ssh.exec_command(cmd)
        exit_code = out.channel.recv_exit_status()
        return exit_code, out.read().decode().strip(), err.read().decode().strip()

    def _wait_optitrack_message(self) -> bool:
        self._optitrack_event.clear()
        return self._optitrack_event.wait(timeout=self._cfg['optitrack']['check_timeout_sec'])

    def _verify_optitrack_header(self) -> None:
        cfg_o = self._cfg['optitrack']
        with self._optitrack_lock:
            msg = self._optitrack_last_msg
        if msg is None:
            raise MissionAbortError("OptiTrack message is None — cannot verify header")

        # Check frame_id
        expected_fid = cfg_o['expected_frame_id']
        if msg.header.frame_id != expected_fid:
            raise MissionAbortError(
                f"OptiTrack frame_id mismatch: got '{msg.header.frame_id}', "
                f"expected '{expected_fid}'")

        # Check freshness
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = now_sec - stamp_sec
        max_age = cfg_o['max_stamp_age_sec']
        if age > max_age:
            raise MissionAbortError(
                f"OptiTrack stamp is stale: age={age:.2f}s > max={max_age}s")

        self._log.info(f"  OptiTrack OK — frame_id='{msg.header.frame_id}', age={age:.3f}s")

    def _wait_drone_state(self, target: int, timeout_sec: float) -> bool:
        """Block until /drone/state is exactly *target*, or *timeout_sec* elapses."""
        deadline = time.monotonic() + timeout_sec
        with self._drone_state_cond:
            while self._drone_state != target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._drone_state_cond.wait(timeout=remaining)
            return True

    def _wait_for_publisher(self, topic: str, timeout_sec: float, label: str) -> None:
        """Block until at least one publisher exists on *topic* or raise."""
        deadline = time.monotonic() + timeout_sec
        while self.count_publishers(topic) == 0:
            if time.monotonic() > deadline:
                raise MissionAbortError(
                    f"{label} not ready: no publisher on '{topic}' after {timeout_sec}s")
            time.sleep(0.5)
        self._log.info(f"  {label} ready — publisher on '{topic}' detected")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 01 — Optitrack bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_01a_check_optitrack(self) -> None:
        self._log.info("╔══ Stage 01.a: Check OptiTrack presence")
        if not self._wait_optitrack_message():
            self._log.warning("  No message — launching optitrack_client and retrying …")
            proc = subprocess.Popen(['ros2', 'run', 'optitrack_client', 'optitrack_client'])
            self._processes['optitrack_client'] = proc
            time.sleep(self._cfg['optitrack']['retry_delay_sec'])
            self._optitrack_event.clear()
            if not self._wait_optitrack_message():
                raise MissionAbortError("OptiTrack did not come up after launching client")
        self._log.info("╚══ Stage 01.a OK: OptiTrack publishing")

    def _stage_01b_optitrack_sanity(self) -> None:
        self._log.info("╔══ Stage 01.b: Sanity-check /optitrack/rigid_body")
        self._verify_optitrack_header()
        self._log.info("╚══ Stage 01.b OK: OptiTrack contents verified")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 02 — Arena map builder bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_02a_configure_background(self) -> None:
        self._log.info("╔══ Stage 02.a: Launch arena_map_builder + set background_path")
        cmd = ['ros2', 'run', 'arena_map_builder', 'build_arena_map_server']
        if self._online_enabled:
            # online_mode must be set at construction (it wires the subscription
            # + start/stop services), so pass it as a launch-time parameter.
            topic = self._cfg['map_builder'].get(
                'online_image_topic', '/camera/image_proc')
            cmd += ['--ros-args',
                    '-p', 'stitch.online_mode:=true',
                    '-p', f'stitch.online_image_topic:={topic}']
            self._log.info(f"  online mode: server will subscribe {topic}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['map_builder'] = proc
        self._log.info(f"  build_arena_map_server launched (pid={proc.pid})")

        # Wait for the action server to be ready
        server_timeout = self._cfg['map_builder']['server_ready_timeout_sec']
        self._log.info(f"  Waiting for action server (up to {server_timeout}s) …")
        if not self._map_action_client.wait_for_server(timeout_sec=server_timeout):
            raise MissionAbortError(
                f"BuildArenaMap action server not available after {server_timeout}s")

        # Set background_path parameter — retry because ros2cli node discovery
        # can lag behind the action server becoming available.
        bg = self._cfg['map_builder']['background_image_path']
        self._log.info(f"  Setting background_path → {bg}")
        result = None
        for attempt in range(1, 6):
            result = subprocess.run(
                ['ros2', 'param', 'set', '/build_arena_map_server',
                 'transfer.background_path', bg],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                break
            self._log.info(
                f"  ros2 param set attempt {attempt}/5 failed "
                f"({result.stderr.strip()!r}), retrying …")
            time.sleep(1.0)
        if result.returncode != 0:
            raise MissionAbortError(
                f"Failed to set background_path after 5 attempts: "
                f"{result.stderr.strip()}")
        self._log.info("╚══ Stage 02.a OK: map_builder ready")

    def _stage_02b_configure_mode(self) -> None:
        self._log.info("╔══ Stage 02.b: Configure online/offline mode")
        mode = 'ONLINE' if self._online_enabled else 'OFFLINE'
        self._log.info(
            f"  map builder mode = {mode}  "
            f"(online stitches live from /camera/image_proc during the flight; "
            f"offline stitches the saved scan.mp4 afterwards)")
        self._log.info(f"╚══ Stage 02.b OK: mode = {mode}")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 03 — Drone routine
    # ════════════════════════════════════════════════════════════════════════

    def _stage_03a_connect_tello_wifi(self) -> None:
        self._log.info("╔══ Stage 03.a: Connect to Tello WiFi")
        cfg_w = self._cfg['tello_wifi']
        iface = cfg_w['interface']
        ssid = cfg_w['ssid']
        scan_retries = int(cfg_w.get('scan_retries', 3))
        connect_timeout = float(cfg_w.get('connect_timeout_sec', 30.0))

        # Scan for the Tello network; retry in case it takes a moment to appear
        self._log.info(f"  Scanning for '{ssid}' on interface {iface} …")
        found = False
        for attempt in range(1, scan_retries + 1):
            result = subprocess.run(
                ['nmcli', 'device', 'wifi', 'list', 'ifname', iface],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                self._log.warning(
                    f"  nmcli scan failed (attempt {attempt}): {result.stderr.strip()}")
            elif ssid in result.stdout:
                found = True
                self._log.info(f"  '{ssid}' visible on scan attempt {attempt}")
                break
            else:
                self._log.info(
                    f"  '{ssid}' not yet visible (attempt {attempt}/{scan_retries}) — retrying …")
                time.sleep(2.0)

        if not found:
            raise MissionAbortError(
                f"Tello WiFi '{ssid}' not found after {scan_retries} scans on {iface}")

        # Connect
        self._log.info(f"  Connecting to '{ssid}' on {iface} …")
        try:
            result = subprocess.run(
                ['sudo', 'nmcli', 'device', 'wifi', 'connect', ssid, 'ifname', iface],
                capture_output=True, text=True,
                timeout=connect_timeout,
            )
        except subprocess.TimeoutExpired:
            raise MissionAbortError(
                f"nmcli connect to '{ssid}' timed out after {connect_timeout}s")

        if result.returncode != 0:
            raise MissionAbortError(
                f"Failed to connect to '{ssid}': "
                f"{result.stderr.strip() or result.stdout.strip()}")

        self._log.info(f"  {result.stdout.strip()}")
        self._log.info("╚══ Stage 03.a OK: Tello WiFi connected")

    def _stage_03b_launch_tello_driver(self) -> None:
        self._log.info("╔══ Stage 03.b: Launch tello_driver")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'tello_driver', 'tello_driver.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['tello_driver'] = proc
        self._log.info(f"  tello_driver launched (pid={proc.pid})")
        delay = float(self._cfg['drone'].get('driver_startup_delay_sec', 15.0))
        self._log.info(f"  Waiting {delay}s for tello_driver to finish configuring …")
        time.sleep(delay)
        self._log.info("╚══ Stage 03.b OK")

    def _stage_03c_drone_preflight(self) -> None:
        self._log.info("╔══ Stage 03.c: Drone preflight checks")
        cfg_d = self._cfg['drone']

        # Camera
        self._log.info(f"  Waiting for {cfg_d['camera_topic']} …")
        if not self._camera_event.wait(timeout=cfg_d['camera_timeout_sec']):
            raise MissionAbortError(
                f"Camera topic {cfg_d['camera_topic']} not active after "
                f"{cfg_d['camera_timeout_sec']}s")
        self._log.info(f"  Camera OK: {cfg_d['camera_topic']} is live")

        # Battery — abort immediately if below threshold (replacing the battery
        # requires a manual restart, so waiting serves no purpose)
        self._log.info(f"  Waiting for battery reading …")
        if not self._battery_event.wait(timeout=cfg_d['battery_timeout_sec']):
            raise MissionAbortError(
                f"Battery topic did not publish within {cfg_d['battery_timeout_sec']}s")
        pct = self._battery_pct
        if pct < cfg_d['battery_min_pct']:
            raise MissionAbortError(
                f"Battery too low: {pct:.1f}% < {cfg_d['battery_min_pct']}% — "
                f"replace battery and restart")
        self._log.info(f"  Battery OK: {pct:.1f}%")

        # Confirm drone state is -1 (before takeoff)
        if self._drone_state is not None and self._drone_state != -1:
            raise MissionAbortError(
                f"Expected drone state -1 (before takeoff), got {self._drone_state}")
        self._log.info("╚══ Stage 03.c OK: Drone is ready for takeoff")

    def _stage_03d_launch_tello_map(self) -> None:
        self._log.info("╔══ Stage 03.d: Launch tello_map (drone take-off + scan)")
        cmd = ['ros2', 'launch', 'tello_pos_control', 'tello_map.launch.py']
        if self._online_enabled:
            # Have the controller republish the processed frames it records so
            # the map-builder can stitch them live.
            topic = self._cfg['map_builder'].get(
                'online_image_topic', '/camera/image_proc')
            cmd += ['publish_processed:=true', f'processed_image_topic:={topic}']
            self._log.info(
                f"  online: controller will publish processed frames → {topic}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['tello_map'] = proc
        self._log.info(f"  tello_map launched (pid={proc.pid})")
        self._log.info("╚══ Stage 03.d OK")

    def _stage_03e_online_start(self) -> None:
        """Tell the map-builder server to begin live stitching (online mode)."""
        self._log.info("╔══ Stage 03.e: Start live stitching (online)")
        timeout = float(self._cfg['map_builder'].get('online_start_timeout_sec', 60.0))
        resp = self._call_service(
            self._online_start_cli, Trigger.Request(), timeout_sec=timeout)
        if not resp.success:
            raise MissionAbortError(f"online_start failed: {resp.message}")
        self._log.info(f"╚══ Stage 03.e OK: {resp.message}")

    def _stage_03f_observe_drone_states(self) -> None:
        self._log.info("╔══ Stage 03.f: Monitor drone state transitions")
        cfg_d = self._cfg['drone']

        state_names = {1: 'Stabilize', 2: 'Trajectory', 3: 'Going Back', 4: 'Landing'}
        timeouts = {
            1: cfg_d['state1_timeout_sec'],
            2: cfg_d['state2_timeout_sec'],
            3: cfg_d['state3_timeout_sec'],
            4: cfg_d['state4_timeout_sec'],
        }

        for state in (1, 2, 3, 4):
            name = state_names[state]
            tmo = timeouts[state]
            self._log.info(f"  Waiting for state {state} ({name}), timeout={tmo}s …")
            if not self._wait_drone_state(state, tmo):
                self._log.error(f"  TIMEOUT waiting for state {state} ({name})")
                self._abort_drone()
                raise MissionAbortError(f"Drone timeout in state {state} ({name})")
            self._log.info(f"  → State {state} ({name}) reached")

        self._log.info("╚══ Stage 03.f OK: Drone mission complete (state 4)")

    def _stage_03g_wait_video_files(self) -> None:
        self._log.info("╔══ Stage 03.g: Wait for video and telemetry files")
        cfg_v = self._cfg['video']
        video_dir = cfg_v['dir']
        video_path = os.path.join(video_dir, cfg_v['video_filename'])
        telemetry_path = os.path.join(video_dir, cfg_v['telemetry_filename'])
        timeout = cfg_v['file_appear_timeout_sec']
        max_age = cfg_v['max_age_sec']

        self._log.info(
            f"  Expecting {video_path!r} and {telemetry_path!r} "
            f"(timeout={timeout}s, max_age={max_age}s)")

        deadline = time.monotonic() + timeout
        while True:
            now = time.time()
            errors = []
            for label, path in ((cfg_v['video_filename'], video_path),
                                (cfg_v['telemetry_filename'], telemetry_path)):
                if not os.path.isfile(path):
                    errors.append(f"{label} not found")
                    continue
                age = now - os.path.getmtime(path)
                if age > max_age:
                    errors.append(f"{label} too old ({age:.0f}s > {max_age}s)")
                    continue
                if os.path.getsize(path) == 0:
                    errors.append(f"{label} is empty")

            if not errors:
                break

            if time.monotonic() >= deadline:
                raise MissionAbortError(
                    f"Video files not ready after {timeout}s: {'; '.join(errors)}")

            self._log.info(f"  Not ready ({'; '.join(errors)}) — retrying in 5s …")
            time.sleep(5.0)

        self._video_path = video_path
        self._telemetry_path = telemetry_path

        now = time.time()
        for label, path in ((cfg_v['video_filename'], video_path),
                            (cfg_v['telemetry_filename'], telemetry_path)):
            size_kb = os.path.getsize(path) / 1024
            age = now - os.path.getmtime(path)
            self._log.info(f"  {label}: {size_kb:.1f} KB, age={age:.0f}s — OK")

        self._log.info("╚══ Stage 03.g OK")

    def _stage_03h_verify_video_integrity(self) -> None:
        self._log.info("╔══ Stage 03.h: Verify scan.mp4 integrity via ffmpeg")
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', self._video_path, '-f', 'null', '-'],
            capture_output=True,
        )
        if result.returncode != 0:
            err = result.stderr.decode().strip()
            raise MissionAbortError(
                f"scan.mp4 failed ffmpeg integrity check: {err}")
        self._log.info("╚══ Stage 03.h OK: scan.mp4 is valid")

    # ── Map-build kickoff: online_stop (if online) + send goal in background ──

    def _online_stop(self) -> None:
        """Tell the server to stop intake and finalize the live stitch."""
        self._log.info("╔══ Map build: stop live stitching + finalize")
        timeout = float(self._cfg['map_builder'].get('online_stop_timeout_sec', 180.0))
        resp = self._call_service(
            self._online_stop_cli, Trigger.Request(), timeout_sec=timeout)
        if not resp.success:
            raise MissionAbortError(f"online_stop failed: {resp.message}")
        self._log.info(f"╚══ Map build: stitch finalized — {resp.message}")

    def _send_map_goal_async(self) -> None:
        """Finalize the live stitch (online) and send BuildArenaMap WITHOUT
        blocking, so transfer+occupancy (and the full stitch in offline) overlaps
        the marker-localizer call. The result is joined later at stage 04.b."""
        if self._online_enabled:
            self._online_stop()

        self._log.info("╔══ Map build: sending BuildArenaMap goal (async)")
        if not self._map_action_client.wait_for_server(timeout_sec=30.0):
            raise MissionAbortError("BuildArenaMap action server not available")

        goal = BuildArenaMap.Goal()
        goal.video_path = self._video_path or ''
        self._map_goal_handle = None
        self._map_result_future = None
        self._map_send_error = None

        accepted_evt = threading.Event()

        def _on_goal(send_future):
            try:
                gh = send_future.result()
                if not gh.accepted:
                    self._map_send_error = "BuildArenaMap goal was rejected"
                else:
                    self._map_goal_handle = gh
                    self._map_result_future = gh.get_result_async()
            except Exception as exc:           # pragma: no cover
                self._map_send_error = f"send_goal failed: {exc}"
            finally:
                accepted_evt.set()

        self._map_action_client.send_goal_async(
            goal, feedback_callback=self._map_action_feedback
        ).add_done_callback(_on_goal)

        if not accepted_evt.wait(timeout=30.0):
            raise MissionAbortError("BuildArenaMap goal not accepted within 30s")
        if self._map_send_error or self._map_result_future is None:
            raise MissionAbortError(self._map_send_error or "BuildArenaMap goal rejected")
        self._log.info("╚══ Map build: running in background")

    def _await_map_result(self) -> "Optional[BuildArenaMap.Result]":
        """Block until the background BuildArenaMap result arrives; return it.

        On any failure (no goal in flight, timeout, or a non-success result) this
        does NOT abort the mission: it logs a warning, clears
        self._last_map_result, and returns None. The fail path (stage 04.c) then
        publishes the border template map and runs frontier exploration."""
        if self._map_result_future is None:
            self._log.warning(
                "  No BuildArenaMap goal in flight — map result unavailable")
            self._last_map_result = None
            return None
        timeout = float(self._cfg['map_builder']['action_timeout_sec'])
        evt = threading.Event()
        self._map_result_future.add_done_callback(lambda _: evt.set())
        if not evt.wait(timeout=timeout):
            self._log.warning(f"  BuildArenaMap timed out after {timeout}s — "
                              f"treating map as unavailable")
            self._last_map_result = None
            return None
        result = self._map_result_future.result().result
        if not result.success:
            self._log.warning(f"  BuildArenaMap failed: {result.message} — "
                              f"treating map as unavailable")
            self._last_map_result = None
            return None
        self._last_map_result = result   # exposed for subclasses (e.g. save_scan)
        self._log.info(
            f"  Map built: {result.map.info.width}×{result.map.info.height} cells, "
            f"{result.n_obstacles} obstacles, "
            f"mean consistency={result.mean_consistency:.3f}")
        gp, ap = result.goal_marker_position, result.amr_marker_position
        self._log.info(
            f"  Marker centres (m from grid origin): "
            f"goal=({gp.x:.3f}, {gp.y:.3f})  amr=({ap.x:.3f}, {ap.y:.3f})")

        # Parse the diagnostic feature vector (parallel name/value arrays) into a
        # dict for the runtime classifier. Empty if the server could not compute
        # it — not fatal here; downstream consumers decide how to handle that.
        self._map_features = {
            str(name): float(val)
            for name, val in zip(result.feature_names, result.feature_values)
        }
        n_feat = len(self._map_features)
        if n_feat:
            self._log.info(f"  Map quality features: {n_feat} received "
                           f"(e.g. blob_count={self._map_features.get('blob_count')}, "
                           f"green_hull_convexity="
                           f"{self._map_features.get('green_hull_convexity')})")
        else:
            self._log.warning("  Map quality features: none returned by server")
        return result

    # ════════════════════════════════════════════════════════════════════════
    # Stage 04 — Aruco localizer
    # ════════════════════════════════════════════════════════════════════════

    def _stage_04_launch_marker_localizer(self) -> None:
        self._log.info("╔══ Stage 04: Launch arena_marker_localizer")
        ws = self._cfg['marker_localizer']['workspace_path']

        # Find the default.yaml for the localizer inside the workspace
        result = subprocess.run(
            ['find', ws, '-name', 'default.yaml',
             '-path', '*/arena_marker_localizer/*',
             '-not', '-path', '*/build/*'],
            capture_output=True, text=True,
        )
        yaml_paths = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        if not yaml_paths:
            raise MissionAbortError(
                f"Could not find arena_marker_localizer default.yaml under {ws}")
        ws_yaml = yaml_paths[0]
        self._log.info(f"  Using YAML: {ws_yaml}")

        proc = subprocess.Popen(
            ['ros2', 'launch', 'arena_marker_localizer',
             'marker_localizer.launch.py', f'params_file:={ws_yaml}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['marker_localizer'] = proc
        self._log.info(f"  marker_localizer launched (pid={proc.pid})")

        # Wait for the service to become available
        server_timeout = self._cfg['marker_localizer']['server_ready_timeout_sec']
        self._log.info(f"  Waiting for /localize_markers service (up to {server_timeout}s) …")
        deadline = time.monotonic() + server_timeout
        while not self._loc_client.wait_for_service(timeout_sec=1.0):
            if time.monotonic() > deadline:
                raise MissionAbortError(
                    f"/localize_markers not available after {server_timeout}s")
        self._log.info("╚══ Stage 04 OK: marker_localizer ready")

    def _stage_04a_call_localize_markers(self) -> List:
        self._log.info("╔══ Stage 04.a: Call /localize_markers")
        req = LocalizeMarkers.Request()
        req.video_path = self._video_path
        req.optitrack_csv = self._telemetry_path

        timeout = self._cfg['marker_localizer']['service_timeout_sec']
        self._log.info(f"  video={self._video_path!r}, csv={self._telemetry_path!r}")
        response = self._call_service(self._loc_client, req, timeout_sec=timeout)

        if not response.success:
            raise MissionAbortError(
                f"Localizer service failed: {response.message}")

        markers = response.markers
        self._log.info(f"  Localized {len(markers)} marker(s):")
        for m in markers:
            self._log.info(
                f"    id={m.id}  x={m.pose_2d.x:.3f} y={m.pose_2d.y:.3f} "
                f"θ={math.degrees(m.pose_2d.theta):.1f}°  n_obs={m.n_observations}")
        self._log.info("╚══ Stage 04.a OK")
        return markers

    def _stage_04b_publish_aruco_poses(self, markers: List) -> None:
        self._log.info("╔══ Stage 04.b: Join map result → publish AMR pose")
        cfg_a = self._cfg['aruco']
        amr_id = cfg_a['amr_marker_id']
        by_id = {int(m.id): m for m in markers}

        # Join the background map-build result (POSITION + OccupancyGrid +
        # diagnostic features). Stored on self._last_map_result for stage 04.c.
        map_result = self._await_map_result()

        # AMR pose: POSITION from the map-builder, ORIENTATION from the localizer.
        # If either is unavailable (including a missing/failed map build), publish
        # an all-NaN pose so alignment_node falls back to the trivial identity
        # transform (no world→odom offset).
        amr_position = map_result.amr_marker_position if map_result is not None else None
        amr_pose, amr_ok = self._build_amr_pose(
            by_id.get(amr_id), amr_position)
        self.secure_publish(self._pub_aruco_amr, amr_pose)
        if amr_ok:
            self._log.info(
                f"  AMR (marker {amr_id}): map pos="
                f"({amr_pose.pose.pose.position.x:.3f}, "
                f"{amr_pose.pose.pose.position.y:.3f}) "
                f"θ={math.degrees(_yaw_from_quat(amr_pose.pose.pose.orientation)):.1f}° "
                f"→ {cfg_a['amr_pose_topic']}")
        else:
            self._log.warning(
                f"  AMR (marker {amr_id}) unavailable (not localized or NaN "
                f"position) → published all-NaN pose; alignment_node will use "
                f"the identity transform")
        self._log.info("╚══ Stage 04.b OK")

    def _stage_04c_classify_and_branch(self, markers: List) -> None:
        """Classify the stitched map; gate the OccupancyGrid + goal pose on it.

        PASS → publish /drone/map and the goal pose; the regular pipeline
               continues using the stitched map.
        FAIL (map unavailable OR classifier reject) → publish a border template
               grid to /drone/map and no goal pose. The planner will use the
               fusion map (stage 11) and frontier exploration is launched after
               stage 11 and triggered at 12.a.
        """
        self._log.info("╔══ Stage 04.c: Map-quality classification")
        cfg_a = self._cfg['aruco']
        goal_id = cfg_a['goal_marker_id']
        by_id = {int(m.id): m for m in markers}
        map_result = self._last_map_result

        map_ok = (map_result is not None) and self._classify_map()
        self._map_failed = not map_ok

        if not self._map_failed:
            # Good map → publish it and the goal pose.
            self._publish_drone_map(map_result.map)
            goal_pose = self._build_goal_pose(
                by_id.get(goal_id), map_result.goal_marker_position)
            if goal_pose is not None:
                self.secure_publish(self._pub_aruco_goal, goal_pose)
                self._log.info(
                    f"  goal (marker {goal_id}): map pos="
                    f"({goal_pose.pose.pose.position.x:.3f}, "
                    f"{goal_pose.pose.pose.position.y:.3f}) "
                    f"θ={math.degrees(_yaw_from_quat(goal_pose.pose.pose.orientation)):.1f}° "
                    f"→ {cfg_a['goal_pose_topic']}")
            else:
                self._log.warning(
                    f"  goal (marker {goal_id}) unavailable → not publishing "
                    f"{cfg_a['goal_pose_topic']}")
            self._log.info("╚══ Stage 04.c OK: map PASSED — continuing with stitched map")
        else:
            # Bad/unavailable map → publish a border template grid to /drone/map
            # (no goal pose). Frontier exploration takes over.
            self._log.warning(
                f"  Map FAILED (unavailable or classifier reject) — publishing "
                f"border template to {self._cfg['map_builder']['drone_map_topic']}, "
                f"no goal pose on {cfg_a['goal_pose_topic']}; frontier "
                f"exploration will take over")
            self._publish_drone_map(self._build_template_grid())
            self._log.info("╚══ Stage 04.c OK: map FAILED — frontier exploration mode")

    def _classify_map(self) -> bool:
        """Return True if the stitched map is good enough to use.

        Uses the diagnostic feature vector returned by BuildArenaMap (stored in
        self._map_features) and the trained RandomForest. Safe defaults: an empty
        feature vector → FAIL (no quality evidence, explore); a classifier
        load/predict error → PASS (don't force exploration on infrastructure
        problems). Disable entirely via config map_classifier.enabled=false.
        """
        cfg_c = self._cfg.get('map_classifier', {})
        if not cfg_c.get('enabled', True):
            self._log.info("  Map classifier disabled → treating map as PASS")
            return True

        if not self._map_features:
            self._log.warning(
                "  No map-quality features in the result → treating map as FAIL "
                "(will explore)")
            return False

        try:
            if self._classifier is None:
                from mission_orchestrator.map_quality_classifier import (
                    MapQualityClassifier)
                model_dir = cfg_c.get('model_dir') or None
                self._classifier = MapQualityClassifier(model_dir)
                self._log.info(f"  Loaded classifier: {self._classifier!r}")
            result = self._classifier.evaluate(self._map_features)
            self._log.info(f"  Classifier → {result}")
            if result.missing:
                self._log.warning(
                    f"  {len(result.missing)} expected feature(s) missing "
                    f"(sentinel-filled): {', '.join(result.missing)}")
            return result.good
        except Exception as exc:
            self._log.error(
                f"  Map classifier error ({exc}) → treating map as PASS "
                f"(continuing with the stitched map)")
            return True

    def _build_amr_pose(self, marker, position):
        """Build the AMR PoseWithCovarianceStamped (POSITION from map-builder,
        ORIENTATION from the localizer), in the `world` frame.

        Returns (pose, available). When the marker was not localized or the
        map-builder position is NaN, returns an all-NaN pose with available=False
        — a "no data" signal that alignment_node turns into the identity
        transform.
        """
        now = self.get_clock().now().to_msg()
        if (marker is None or position is None
                or math.isnan(position.x) or math.isnan(position.y)):
            nan = float('nan')
            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = now
            pose.pose.pose.position = Point(x=nan, y=nan, z=nan)
            pose.pose.pose.orientation = Quaternion(x=nan, y=nan, z=nan, w=nan)
            return pose, False

        pose = marker.pose_with_covariance   # PoseWithCovarianceStamped (localizer)
        pose.pose.pose.position.x = float(position.x)
        pose.pose.pose.position.y = float(position.y)
        pose.pose.pose.position.z = 0.0
        pose.header.frame_id = 'world'
        pose.header.stamp = now
        return pose, True

    def _build_goal_pose(self, marker, position):
        """Build the goal PoseWithCovarianceStamped, or None if unavailable.

        POSITION is taken from the map-builder result; ORIENTATION, covariance,
        and header come from the localizer's `marker.pose_with_covariance`. The
        localizer's map frame and the grid frame share the same origin, so only
        the position is swapped. Returns None when the marker was not localized or
        the map-builder could not place it (NaN position).
        """
        if (marker is None or position is None
                or math.isnan(position.x) or math.isnan(position.y)):
            return None

        pose = marker.pose_with_covariance   # PoseWithCovarianceStamped (localizer)
        pose.pose.pose.position.x = float(position.x)
        pose.pose.pose.position.y = float(position.y)
        pose.pose.pose.position.z = 0.0
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        return pose

    def _publish_drone_map(self, grid: OccupancyGrid) -> None:
        """Publish the map-builder OccupancyGrid to /drone/map (stage 02.c)."""
        cfg_mb = self._cfg['map_builder']
        grid.header.frame_id = 'world'
        grid.header.stamp = self.get_clock().now().to_msg()

        self.secure_publish(self._pub_drone_map, grid)
        self._log.info(
            f"  Published {grid.info.width}×{grid.info.height} OccupancyGrid "
            f"to {cfg_mb['drone_map_topic']}")

        # Confirm our own subscriber fires (latched; same process)
        if not self._drone_map_event.wait(timeout=5.0):
            self._log.warning("  /drone/map subscriber did not echo back within 5s (may be ok)")

    def _build_template_grid(self) -> OccupancyGrid:
        """Build a bordered empty OccupancyGrid from map_builder.template_grid.

        Free everywhere except an outer ring of `border_cells` wall cells. Used as
        the fail-mode /drone/map when the stitched map is unavailable/rejected."""
        cfg_t = self._cfg['map_builder']['template_grid']
        res = float(cfg_t['resolution_m_per_cell'])
        W = round(float(cfg_t['arena_width_m']) / res)
        H = round(float(cfg_t['arena_height_m']) / res)
        wall_occ = int(cfg_t['wall_occ'])
        free_occ = int(cfg_t['free_occ'])
        border = int(cfg_t['border_cells'])

        data = [free_occ] * (W * H)
        for y in range(H):
            for x in range(W):
                if x < border or x >= W - border or y < border or y >= H - border:
                    data[y * W + x] = wall_occ

        grid = OccupancyGrid()
        grid.header.frame_id = str(cfg_t['frame_id'])
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = res
        grid.info.width = W
        grid.info.height = H
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data
        return grid

    # ════════════════════════════════════════════════════════════════════════
    # Stage 05 — Isaac ROS Visual SLAM (cuSLAM) bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_05a_verify_realsense(self) -> None:
        self._log.info("╔══ Stage 05.a: Verify Intel RealSense D435i")
        cfg_v = self._cfg.get('vslam', {})
        usb_id = str(cfg_v.get('realsense_usb_id', '8086:0b3a')).lower()
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        out = result.stdout.lower()
        if usb_id in out or 'realsense' in out:
            self._log.info("╚══ Stage 05.a OK: RealSense detected on USB")
        else:
            raise MissionAbortError(
                f"Intel RealSense D435i (usb {usb_id}) not found in lsusb — "
                f"check it is plugged in")

    def _stage_05b_start_vslam(self) -> None:
        # Covers spec 05.b (start container) + 05.c (launch visual SLAM).
        self._log.info("╔══ Stage 05.b: Start container + launch Isaac ROS VSLAM")
        cfg_v = self._cfg.get('vslam', {})
        script = cfg_v.get('start_script')
        if not script or not os.path.isfile(script):
            raise MissionAbortError(
                f"vslam.start_script not found: {script!r}")
        timeout = float(cfg_v.get('start_timeout_sec', 120.0))
        result = subprocess.run(
            ['bash', script], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise MissionAbortError(
                f"start_vslam.sh failed (rc={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}")
        self._log.info(f"  {result.stdout.strip()}")
        self._log.info("╚══ Stage 05.b OK: VSLAM container + launch started")

    def _stage_05c_check_vslam_odometry(self) -> None:
        self._log.info("╔══ Stage 05.c: Check /visual_slam/tracking/odometry")
        cfg_v = self._cfg.get('vslam', {})
        topic = cfg_v.get('odometry_topic', '/visual_slam/tracking/odometry')
        timeout = float(cfg_v.get('odometry_timeout_sec', 30.0))
        self._vslam_odom_event.clear()
        self._log.info(f"  Waiting for {topic} (up to {timeout:.0f}s) …")
        if not self._vslam_odom_event.wait(timeout=timeout):
            raise MissionAbortError(
                f"No message on {topic} within {timeout:.0f}s — VSLAM not tracking")
        self._log.info("╚══ Stage 05.c OK: VSLAM odometry live")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 06 — Rasp bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_06a_ping(self) -> None:
        self._log.info("╔══ Stage 06.a: Ping Raspberry Pi")
        cfg_r = self._cfg['rasp']
        ip = cfg_r['ip']
        cmd = ['ping', '-c', str(cfg_r['ping_count']),
               '-W', str(int(cfg_r['ping_timeout_sec'])), ip]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise MissionAbortError(f"Raspberry Pi {ip} is not reachable")
        self._log.info(f"╚══ Stage 06.a OK: Raspberry Pi {ip} reachable")

    def _stage_06b_ssh_connect(self) -> None:
        self._log.info("╔══ Stage 06.b: SSH connect to Raspberry Pi")
        cfg_r = self._cfg['rasp']
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=cfg_r['ip'],
            username=cfg_r['user'],
            password=cfg_r['password'],
            timeout=cfg_r['ssh_connect_timeout_sec'],
        )
        self._ssh = client
        self._log.info(f"╚══ Stage 06.b OK: SSH connected to {cfg_r['user']}@{cfg_r['ip']}")

    def _stage_06c_launch_amr(self) -> None:
        self._log.info("╔══ Stage 06.c: Start AMR bringup service on Raspberry Pi")
        cfg_r = self._cfg['rasp']
        svc = cfg_r['amr_service']
        timeout = cfg_r['amr_service_start_timeout_sec']

        # Start the service (idempotent — fine if already running)
        rc, _, stderr = self._ssh_run(f'systemctl start {svc}')
        if rc != 0:
            raise MissionAbortError(
                f"systemctl start {svc} failed (rc={rc}): {stderr}")
        self._log.info(f"  systemctl start {svc} → OK")

        # Poll until active or timeout
        self._log.info(f"  Waiting for {svc} to become active (up to {timeout}s) …")
        deadline = time.monotonic() + timeout
        while True:
            _, state, _ = self._ssh_run(f'systemctl is-active {svc}')
            if state == 'active':
                break
            if time.monotonic() > deadline:
                # Grab journal tail for diagnostics
                _, journal, _ = self._ssh_run(
                    f'journalctl -u {svc} -n 20 --no-pager')
                raise MissionAbortError(
                    f"{svc} did not become active within {timeout}s "
                    f"(state={state!r}).\nJournal tail:\n{journal}")
            time.sleep(1.0)

        self._log.info(f"╚══ Stage 06.c OK: {svc} is active")

    def _stage_06d_wait_imu_ready(self) -> None:
        n = self._cfg['imu']['message_count']
        timeout = self._cfg['imu']['timeout_sec']
        self._log.info(f"╔══ Stage 06.d: Waiting for {n} messages on {self._cfg['imu']['topic']}")
        if not self._imu_ready_event.wait(timeout=timeout):
            raise MissionAbortError(
                f"IMU did not publish {n} messages within {timeout}s "
                f"(received {self._imu_msg_count})")
        self._log.info(f"╚══ Stage 06.d OK: IMU ready ({self._imu_msg_count} messages received)")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 07 — Emergency stop bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_07a_emergency_stop(self) -> None:
        self._log.info("╔══ Stage 07.a: Launch emergency_stop node")
        cfg_es = self._cfg.get('emergency_stop', {})
        topic = cfg_es.get('topic', '/amr/emergency_stop')
        n_samples = int(cfg_es.get('check_count', 10))
        startup_timeout = float(cfg_es.get('startup_timeout_sec', 15.0))

        proc = subprocess.Popen(
            ['ros2', 'launch', 'emergency_stop', 'emergency_stop.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['emergency_stop'] = proc
        self._log.info(f"  emergency_stop launched (pid={proc.pid})")

        # Reset collection state before we start counting messages
        with self._estop_lock:
            self._estop_count = 0
            self._estop_triggered = False
        self._estop_event.clear()

        # At 10 Hz, n_samples messages arrive in n_samples/10 s; add startup margin
        total_timeout = startup_timeout + n_samples / 10.0 + 2.0
        self._log.info(
            f"  Waiting for {n_samples} messages on {topic} "
            f"(timeout={total_timeout:.0f}s) …")

        if not self._estop_event.wait(timeout=total_timeout):
            raise MissionAbortError(
                f"emergency_stop: received only {self._estop_count}/{n_samples} "
                f"messages on {topic} within {total_timeout:.0f}s")

        if self._estop_triggered:
            raise MissionAbortError(
                f"emergency_stop: ACTIVE signal detected on {topic} — "
                "check AMR safety state before proceeding")

        self._log.info(
            f"╚══ Stage 07.a OK: {n_samples} messages on {topic} all False")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 08 — AMR Aruco localizer
    # ════════════════════════════════════════════════════════════════════════

    def _stage_08_amr_localizer(self) -> None:
        """Launch the AMR ArUco localizer (/aruco_pose → EKF).

        Camera source follows the vslam.enabled flag: with VSLAM disabled the
        localizer uses the RealSense color stream (use_realsense:=true,
        defaults); with VSLAM enabled it consumes the VSLAM RealSense IR stream
        (use_realsense:=false, IR image/info topics + IR optical frame)."""
        self._log.info("╔══ Stage 08: Launch AMR aruco_localizer")
        cfg_al = self._cfg.get('amr_localizer', {})
        cmd = ['ros2', 'launch', 'aruco_localizer', 'aruco_localizer.launch.py']
        if self._cfg.get('vslam', {}).get('enabled', False):
            cmd += [
                'use_realsense:=false',
                f"image_topic:={cfg_al.get('vslam_image_topic', '/camera/infra1/image_rect_raw')}",
                f"camera_info_topic:={cfg_al.get('vslam_camera_info_topic', '/camera/infra1/camera_info')}",
                f"camera_frame:={cfg_al.get('vslam_camera_frame', 'camera_infra1_optical_frame')}",
            ]
            self._log.info("  VSLAM enabled → localizer uses the VSLAM RealSense IR stream")
        else:
            cmd += ['use_realsense:=true']
            self._log.info("  VSLAM disabled → localizer uses its own RealSense color stream")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['amr_localizer'] = proc
        self._log.info(f"  aruco_localizer launched (pid={proc.pid})")
        pose_topic = cfg_al.get('pose_topic', '/aruco_pose')
        timeout = float(cfg_al.get('ready_timeout_sec', 30.0))
        self._wait_for_publisher(pose_topic, timeout, 'amr_localizer')
        self._log.info("╚══ Stage 08 OK")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 09 — Mapping bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_09a_launch_oradar(self) -> None:
        self._log.info("╔══ Stage 09.a: Launch oradar lidar")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'oradar_lidar', 'ms200_scan.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['oradar'] = proc
        self._log.info(f"  oradar launched (pid={proc.pid})")
        cfg_or = self._cfg['oradar']
        self._wait_for_publisher(
            cfg_or['scan_topic'], cfg_or['ready_timeout_sec'], 'oradar')
        self._log.info("╚══ Stage 09.a OK")

    def _stage_09b_publish_static_tf(self) -> None:
        self._log.info("╔══ Stage 09.b: Launch alignment_node (world->odom tf)")
        proc = subprocess.Popen(
            ['ros2', 'run', 'amr_drone_nav', 'alignment_node'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['world_odom_tf'] = proc
        self._log.info(f"╚══ Stage 09.b OK: alignment_node launched (pid={proc.pid})")

    def _stage_09c_amr_mapper(self) -> None:
        self._log.info("╔══ Stage 09.c: Launch odom-based mapper (no SLAM)")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'world_mapper', 'mapper.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['amr_mapper'] = proc
        self._log.info(f"╚══ Stage 09.c OK: amr_mapper launched (pid={proc.pid})")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 10 — Map fusion
    # ════════════════════════════════════════════════════════════════════════

    def _stage_10_map_fusion(self) -> None:
        # Fire-and-forget: launch the fusion node and move on. We do NOT wait on a
        # readiness topic — the node only emits its fused grid once a /drone/map is
        # available, so there is nothing meaningful to block on here.
        self._log.info("╔══ Stage 10: Launch map fusion")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'fusion', 'fusion.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['map_fusion'] = proc
        self._log.info(f"  fusion launched (pid={proc.pid})")
        self._log.info("╚══ Stage 10 OK")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 11 — Trajectory planner bringup
    # ════════════════════════════════════════════════════════════════════════

    def _stage_11_trajectory_planner(self) -> None:
        # The planner's map_topic depends on the 04.c decision: the good stitched
        # map (/drone/map) on PASS, or the fusion map (/fusion/map) on FAIL since
        # the stitched map was dumped and frontier exploration takes over.
        if self._map_failed:
            map_topic = self._cfg.get('frontier', {}).get(
                'fusion_map_topic', '/fusion/map')
            self._log.info(
                f"╔══ Stage 11: Launch trajectory_planner (map FAILED → {map_topic})")
        else:
            map_topic = self._cfg['map_builder'].get('drone_map_topic', '/drone/map')
            self._log.info("╔══ Stage 11: Launch trajectory_planner")

        proc = subprocess.Popen(
            ['ros2', 'launch', 'trajectory_planner', 'planner_launch.py',
             f'map_topic:={map_topic}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['trajectory_planner'] = proc
        self._log.info(f"  trajectory_planner launched (pid={proc.pid}) "
                       f"map_topic={map_topic}")
        cfg_tp = self._cfg['trajectory_planner']
        self._wait_for_publisher(
            cfg_tp['ready_topic'], cfg_tp['ready_timeout_sec'], 'trajectory_planner')
        self._log.info("╚══ Stage 11 OK")

    # ── Frontier-exploration fallback (stage 11.b + 12.a) ─────────────────────

    def _stage_11b_launch_frontier(self) -> None:
        """Launch the frontier-exploration fallback stack (FAIL mode only).

        Deferred to here (after stage 11) so its prerequisites — SLAM, EKF, the
        D435i driver, the AMR controller, and the trajectory planner — are
        already up. The nodes idle until the 12.a trigger on
        /map_fail_fallback/start. astar_planner2 + spline_follower are NOT started
        here; they come from stage 11 (the single planner instance)."""
        cfg_f = self._cfg.get('frontier', {})
        goal_id = self._cfg['aruco']['goal_marker_id']
        self._log.info("╔══ Stage 11.b: Launch frontier-exploration fallback")
        cmd = [
            'ros2', 'launch', 'frontier_explorer', 'frontier_explorer_launch.py',
            f"odom_topic:={cfg_f.get('odom_topic', '/amr/ekf/odom')}",
            f"world_frame:={cfg_f.get('world_frame', 'world')}",
            f"slam_map_topic:={cfg_f.get('slam_map_topic', '/slam/map')}",
            f"image_topic:={cfg_f.get('image_topic', '/camera/camera/color/image_raw')}",
            f"camera_info_topic:={cfg_f.get('camera_info_topic', '/camera/camera/color/camera_info')}",
            f"target_marker_id:={goal_id}",
            f"marker_size_m:={cfg_f.get('marker_size_m', 0.13)}",
            f"aruco_dict:={cfg_f.get('aruco_dict', 'DICT_4X4_50')}",
            f"rviz:={'true' if cfg_f.get('rviz', False) else 'false'}",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._processes['frontier_exploration'] = proc
        self._log.info(
            f"  frontier_explorer launched (pid={proc.pid}) — idle until 12.a")
        self._log.info("╚══ Stage 11.b OK")

    def _stage_12a_start_frontier(self) -> None:
        """Trigger frontier exploration by publishing True on the start topic.

        Sent once the rest of the bringup is complete. The subscriber uses
        VOLATILE QoS (no replay), so we wait for it to appear before publishing
        and send a few times to be safe."""
        if not self._map_failed:
            return
        cfg_f = self._cfg.get('frontier', {})
        topic = cfg_f.get('start_topic', '/map_fail_fallback/start')
        timeout = float(cfg_f.get('start_ready_timeout_sec', 30.0))
        self._log.info(f"╔══ Stage 12.a: Start frontier exploration → {topic}")

        deadline = time.monotonic() + timeout
        while self._pub_map_fail_start.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                self._log.warning(
                    f"  No subscriber on {topic} after {timeout:.0f}s — "
                    f"publishing anyway")
                break
            time.sleep(0.5)

        msg = Bool()
        msg.data = True
        for _ in range(3):
            self.secure_publish(self._pub_map_fail_start, msg)
            time.sleep(0.2)
        self._log.info("╚══ Stage 12.a OK: frontier exploration triggered")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 12 — Observer
    # ════════════════════════════════════════════════════════════════════════

    def _stage_12_observer(self) -> None:
        self._log.info("╔══ Stage 12: Enter observer mode")
        self._start_observer()
        self._log.info("╚══ Stage 12 OK: observing")

    # ── Observer implementation ──────────────────────────────────────────────

    def _start_observer(self) -> None:
        cfg_obs = self._cfg.get('observer', {})
        rate_hz = float(cfg_obs.get('rate_hz', 0.5))
        topic_list = list(cfg_obs.get('topics', list(_OBSERVER_REGISTRY.keys())))

        valid   = [t for t in topic_list if t in _OBSERVER_REGISTRY]
        unknown = [t for t in topic_list if t not in _OBSERVER_REGISTRY]
        if unknown:
            self._log.warning(f"[observer] Unknown topics (skipped): {unknown}")
        if not valid:
            self._log.warning("[observer] No recognisable topics — observer disabled.")
            return

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        default = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        for topic in valid:
            qos_key, _, msg_type, _ = _OBSERVER_REGISTRY[topic]
            qos = latched if qos_key == 'latched' else default
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._observer_cb(t, msg),
                qos,
            )

        self._observer_topics = valid
        self.create_timer(1.0 / rate_hz, self._observer_tick)
        self._log.info(
            f"[observer] Started — {len(valid)} topic(s) at {rate_hz} Hz.")

    def _observer_cb(self, topic: str, msg) -> None:
        with self._observer_lock:
            self._observer_cache[topic] = msg

    def _observer_tick(self) -> None:
        with self._observer_lock:
            cache = dict(self._observer_cache)

        if not self._observer_topics:
            return

        label_w = max(len(_OBSERVER_REGISTRY[t][1]) for t in self._observer_topics)
        lines = ['[observer]']
        for topic in self._observer_topics:
            _, label, _, fmt_fn = _OBSERVER_REGISTRY[topic]
            msg = cache.get(topic)
            if msg is None:
                content = '(no data yet)'
            else:
                try:
                    content = fmt_fn(msg)
                except Exception as exc:
                    content = f'(format error: {exc})'
            lines.append(f"  {label:<{label_w}}: {content}")
        self._log.info('\n'.join(lines))

    # ────────────────────────────────────────────────────────────────────────
    # Main orchestration loop
    # ────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._log.info("━━━━━━━━━━━━━━━━  MISSION START  ━━━━━━━━━━━━━━━━")
        self._start_rosbag()
        try:
            # ── 01 Optitrack bringup ──
            self._log.info("━━━━━━  Stage 01: Optitrack bringup  ━━━━━━")
            self._stage_01a_check_optitrack()
            self._stage_01b_optitrack_sanity()

            # ── 02 Arena map builder bringup ──
            self._log.info("━━━━━━  Stage 02: Arena map builder bringup  ━━━━━━")
            self._stage_02a_configure_background()
            self._stage_02b_configure_mode()

            # ── 03 Drone routine ──
            self._log.info("━━━━━━  Stage 03: Drone routine  ━━━━━━")
            self._stage_03a_connect_tello_wifi()
            self._stage_03b_launch_tello_driver()
            self._stage_03c_drone_preflight()
            self._stage_03d_launch_tello_map()
            if self._online_enabled:
                self._stage_03e_online_start()
            self._stage_03f_observe_drone_states()
            self._stage_03g_wait_video_files()
            self._stage_03h_verify_video_integrity()

            # Drone has landed → finalize the live stitch (online) and kick off
            # the BuildArenaMap goal in the background. Joined at 04.b.
            self._send_map_goal_async()

            # ── 04 Aruco localizer (runs concurrently with the map build) ──
            self._log.info("━━━━━━  Stage 04: Aruco localizer  ━━━━━━")
            self._stage_04_launch_marker_localizer()
            markers = self._stage_04a_call_localize_markers()
            self._stage_04b_publish_aruco_poses(markers)
            self._stage_04c_classify_and_branch(markers)

            # ── 05 Isaac ROS Visual SLAM bringup (gated on vslam.enabled) ──
            if self._cfg.get('vslam', {}).get('enabled', False):
                self._log.info("━━━━━━  Stage 05: Isaac ROS Visual SLAM bringup  ━━━━━━")
                self._stage_05a_verify_realsense()
                self._stage_05b_start_vslam()
                self._stage_05c_check_vslam_odometry()
            else:
                self._log.info("Stage 05 VSLAM disabled — skipping")

            # ── 06 Rasp bringup ──
            self._log.info("━━━━━━  Stage 06: Rasp bringup  ━━━━━━")
            self._stage_06a_ping()
            self._stage_06b_ssh_connect()
            self._stage_06c_launch_amr()
            self._stage_06d_wait_imu_ready()

            # ── 07 Emergency stop bringup (opt-in via emergency_stop.enabled) ──
            if self._cfg.get('emergency_stop', {}).get('enabled', False):
                self._log.info("━━━━━━  Stage 07: Emergency stop bringup  ━━━━━━")
                self._stage_07a_emergency_stop()
            else:
                self._log.info("Stage 07 emergency stop disabled — skipping")

            # ── 08 AMR Aruco localizer ──
            self._log.info("━━━━━━  Stage 08: AMR Aruco localizer  ━━━━━━")
            self._stage_08_amr_localizer()

            # ── 09 Mapping bringup ──
            self._log.info("━━━━━━  Stage 09: Mapping bringup  ━━━━━━")
            self._stage_09a_launch_oradar()
            self._stage_09b_publish_static_tf()
            self._stage_09c_amr_mapper()

            # ── 10 Map fusion ──
            self._log.info("━━━━━━  Stage 10: Map fusion  ━━━━━━")
            self._stage_10_map_fusion()

            # ── 11 Trajectory planner bringup ──
            self._log.info("━━━━━━  Stage 11: Trajectory planner bringup  ━━━━━━")
            self._stage_11_trajectory_planner()
            # 11.b (map FAILED only): bring up the frontier-exploration stack now
            # that its prerequisites are running; it idles until the 12.a trigger.
            if self._map_failed:
                self._stage_11b_launch_frontier()

            # ── 12 Observer ──
            # 12.a: if the map failed, start frontier exploration now that the
            # rest of the bringup is complete.
            self._stage_12a_start_frontier()
            self._mission_complete = True
            self._log.info(
                "━━━━━━  MISSION ORCHESTRATION COMPLETE  ━━━━━━\n"
                "mapper and trajectory_planner are now operating autonomously.\n"
                "Press Ctrl+C to exit.")
            self._stage_12_observer()

        except MissionAbortError as exc:
            self._log.error(f"MISSION ABORTED: {exc}")
            self._abort()
        except Exception as exc:
            self._log.error(f"Unexpected error: {exc}", exc_info=True)
            self._abort()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)

    # Resolve config file path from ROS parameter
    tmp = rclpy.create_node('_cfg_reader')
    tmp.declare_parameter('config_file', '')
    config_path = tmp.get_parameter('config_file').value
    tmp.destroy_node()

    if not config_path:
        try:
            from ament_index_python.packages import get_package_share_directory
            config_path = os.path.join(
                get_package_share_directory('mission_orchestrator'),
                'config', 'orchestrator_params.yaml',
            )
        except Exception:
            pass

    if not config_path or not os.path.isfile(config_path):
        print(
            f"ERROR: config_file not found: {config_path!r}\n"
            "Pass it with:  --ros-args -p config_file:=/abs/path/orchestrator_params.yaml",
            file=sys.stderr,
        )
        rclpy.shutdown()
        sys.exit(1)

    node = MissionOrchestratorNode(config_path)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # Spin ROS 2 in a background thread; main thread drives the orchestration
    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        # After orchestration completes (or aborts), stay alive as observer
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._mission_complete:
            node._abort()
        node._stop_rosbag()   # always stop recording on exit
        node._teardown_ssh()  # always stop AMR service on exit
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
