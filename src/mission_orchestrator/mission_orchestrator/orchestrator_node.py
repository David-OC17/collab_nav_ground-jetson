"""
mission_orchestrator.orchestrator_node
────────────────────────────────────────────────────────────────────────────
End-to-end mission sequencer for the collab_nav ground-robot + drone system.

Stages
──────
 01  Ping Raspberry Pi
 02  SSH connect to Raspberry Pi
 03  Start amr_bringup systemd service on Raspberry Pi; verify active
 04  Wait for /imu/data_raw to publish 200 messages (IMU running at 100 Hz)
 05  Check /optitrack/rigid_body presence + header; launch client if absent
 05b Connect to Tello WiFi (nmcli scan + connect on wlx14ebb67dae0b)
 06  Launch tello_driver
 07  Drone preflight: verify /camera/image_raw live, /battery_state ≥ min %
 08  Launch tello_map (drone takes off and executes scanning routine)
 09  Observe drone state transitions 1→2→3→4 with per-stage timeouts
10  Wait for /drone/video_filename and /drone/telemetry_filename topic messages
11  Verify scan.mp4 integrity via ffmpeg
12  Launch trajectory_planner
13  Launch map_fusion
14  Launch oradar lidar
15  Launch arena_marker_localizer service node + wait for readiness
16  Call /localize_markers service → parse marker poses
17  Publish /aruco/amr/pose and /aruco/goal/pose as PoseWithCovarianceStamped
18  Launch arena_map_builder server + set background_path parameter
19  Send BuildArenaMap action goal → wait for result
20  Publish the resulting OccupancyGrid to /drone/map

From stage 20 onward the orchestrator only observes; map_fusion and
trajectory_planner operate autonomously.
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

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import BatteryState, Image, Imu
from std_msgs.msg import Int32, String

from arena_map_builder_msgs.action import BuildArenaMap
from arena_marker_localizer_interfaces.srv import LocalizeMarkers


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
# Orchestrator node
# ─────────────────────────────────────────────────────────────────────────────

class MissionOrchestratorNode(Node):
    """ROS 2 node that sequentially executes all mission stages."""

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, config_path: str) -> None:
        super().__init__('mission_orchestrator')

        self._cfg: dict = self._load_config(config_path)
        self._log: logging.Logger = self._setup_logging()

        # Subprocess handles
        self._processes: Dict[str, subprocess.Popen] = {}

        # SSH state
        self._ssh: Optional[paramiko.SSHClient] = None

        # Mission state flags
        self._mission_complete = False
        self._drone_aborted = False

        # ── ROS 2 sync primitives ──
        self._imu_ready_event = threading.Event()
        self._imu_msg_count: int = 0

        self._optitrack_event = threading.Event()
        self._optitrack_last_msg: Optional[PoseStamped] = None
        self._optitrack_lock = threading.Lock()

        self._drone_state_events: Dict[int, threading.Event] = {
            s: threading.Event() for s in (-1, 0, 1, 2, 3, 4)
        }
        self._drone_state: Optional[int] = None

        self._camera_event = threading.Event()
        self._battery_event = threading.Event()
        self._battery_pct: Optional[float] = None

        self._video_filename_event = threading.Event()
        self._telemetry_filename_event = threading.Event()
        self._video_path: Optional[str] = None
        self._telemetry_path: Optional[str] = None

        self._drone_map_event = threading.Event()

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

        self.create_subscription(Imu,
            cfg['imu']['topic'], self._imu_cb, 10)
        self.create_subscription(PoseStamped,
            cfg['optitrack']['topic'], self._optitrack_cb, qos)
        self.create_subscription(Int32,
            cfg['drone']['state_topic'], self._drone_state_cb, 10)
        self.create_subscription(Image,
            cfg['drone']['camera_topic'], self._camera_cb, best_effort)
        self.create_subscription(BatteryState,
            cfg['drone']['battery_topic'], self._battery_cb, 10)
        self.create_subscription(String,
            cfg['drone']['video_filename_topic'], self._video_filename_cb, 10)
        self.create_subscription(String,
            cfg['drone']['telemetry_filename_topic'], self._telemetry_filename_cb, 10)
        self.create_subscription(OccupancyGrid,
            cfg['map_builder']['drone_map_topic'], self._drone_map_cb, latched)

        self._pub_aruco_amr = self.create_publisher(
            PoseWithCovarianceStamped, cfg['aruco']['amr_pose_topic'], latched)
        self._pub_aruco_goal = self.create_publisher(
            PoseWithCovarianceStamped, cfg['aruco']['goal_pose_topic'], latched)
        self._pub_drone_map = self.create_publisher(
            OccupancyGrid, cfg['map_builder']['drone_map_topic'], latched)

        self._loc_client = self.create_client(
            LocalizeMarkers, cfg['marker_localizer']['service_name'])
        self._map_action_client = ActionClient(
            self, BuildArenaMap, cfg['map_builder']['action_name'])

    # ────────────────────────────────────────────────────────────────────────
    # ROS 2 callbacks
    # ────────────────────────────────────────────────────────────────────────

    def _imu_cb(self, _msg: Imu) -> None:
        self._imu_msg_count += 1
        if (self._imu_msg_count >= self._cfg['imu']['message_count']
                and not self._imu_ready_event.is_set()):
            self._imu_ready_event.set()

    def _optitrack_cb(self, msg: PoseStamped) -> None:
        with self._optitrack_lock:
            self._optitrack_last_msg = msg
        self._optitrack_event.set()

    def _drone_state_cb(self, msg: Int32) -> None:
        state = int(msg.data)
        self._drone_state = state
        evt = self._drone_state_events.get(state)
        if evt is not None:
            evt.set()
        self._log.debug(f"Drone state → {state}")

    def _camera_cb(self, _msg: Image) -> None:
        self._camera_event.set()

    def _battery_cb(self, msg: BatteryState) -> None:
        self._battery_pct = float(msg.percentage)
        if self._battery_pct >= self._cfg['drone']['battery_min_pct']:
            self._battery_event.set()

    def _video_filename_cb(self, msg: String) -> None:
        self._video_path = msg.data
        self._video_filename_event.set()
        self._log.debug(f"Received video_filename: {msg.data!r}")

    def _telemetry_filename_cb(self, msg: String) -> None:
        self._telemetry_path = msg.data
        self._telemetry_filename_event.set()
        self._log.debug(f"Received telemetry_filename: {msg.data!r}")

    def _drone_map_cb(self, _msg: OccupancyGrid) -> None:
        self._drone_map_event.set()

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

    def _call_action(self, goal, timeout_sec: float):
        """Send a BuildArenaMap action goal and wait for the result."""
        deadline = time.monotonic() + timeout_sec

        # Wait for server
        server_timeout = 30.0
        if not self._map_action_client.wait_for_server(timeout_sec=server_timeout):
            raise MissionAbortError(
                f"BuildArenaMap action server not available after {server_timeout}s")

        # Send goal
        goal_evt = threading.Event()
        goal_future = self._map_action_client.send_goal_async(
            goal,
            feedback_callback=self._map_action_feedback,
        )
        goal_future.add_done_callback(lambda _: goal_evt.set())
        if not goal_evt.wait(timeout=30.0):
            raise MissionAbortError("BuildArenaMap goal not accepted within 30s")

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            raise MissionAbortError("BuildArenaMap goal was rejected")

        # Wait for result
        result_evt = threading.Event()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda _: result_evt.set())
        remaining = max(1.0, deadline - time.monotonic())
        if not result_evt.wait(timeout=remaining):
            raise MissionAbortError(
                f"BuildArenaMap action timed out after {timeout_sec}s")
        return result_future.result().result

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

    def _abort(self) -> None:
        """Kill all processes and close SSH.  Idempotent."""
        self._log.error("══════ MISSION ABORT ══════")
        self._abort_drone()
        for name, proc in list(self._processes.items()):
            _kill_proc(proc, name, self._log)
        self._teardown_ssh()
        self._log.error("══════ ABORT COMPLETE ══════")

    # ────────────────────────────────────────────────────────────────────────
    # Stages
    # ────────────────────────────────────────────────────────────────────────

    def _stage_01_ping(self) -> None:
        self._log.info("╔══ Stage 01: Ping Raspberry Pi")
        cfg_r = self._cfg['rasp']
        ip = cfg_r['ip']
        cmd = ['ping', '-c', str(cfg_r['ping_count']),
               '-W', str(int(cfg_r['ping_timeout_sec'])), ip]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise MissionAbortError(f"Raspberry Pi {ip} is not reachable")
        self._log.info(f"╚══ Stage 01 OK: Raspberry Pi {ip} reachable")

    def _stage_02_ssh_connect(self) -> None:
        self._log.info("╔══ Stage 02: SSH connect to Raspberry Pi")
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
        self._log.info(f"╚══ Stage 02 OK: SSH connected to {cfg_r['user']}@{cfg_r['ip']}")

    def _ssh_run(self, cmd: str) -> tuple[int, str, str]:
        """Run a command over SSH; return (exit_code, stdout, stderr)."""
        _in, out, err = self._ssh.exec_command(cmd)
        exit_code = out.channel.recv_exit_status()
        return exit_code, out.read().decode().strip(), err.read().decode().strip()

    def _stage_03_launch_amr(self) -> None:
        self._log.info("╔══ Stage 03: Start AMR bringup service on Raspberry Pi")
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

        self._log.info(f"╚══ Stage 03 OK: {svc} is active")

    def _stage_04_wait_imu_ready(self) -> None:
        n = self._cfg['imu']['message_count']
        timeout = self._cfg['imu']['timeout_sec']
        self._log.info(f"╔══ Stage 04: Waiting for {n} messages on {self._cfg['imu']['topic']}")
        if not self._imu_ready_event.wait(timeout=timeout):
            raise MissionAbortError(
                f"IMU did not publish {n} messages within {timeout}s "
                f"(received {self._imu_msg_count})")
        self._log.info(f"╚══ Stage 04 OK: IMU ready ({self._imu_msg_count} messages received)")

    def _stage_05_check_optitrack(self) -> None:
        self._log.info("╔══ Stage 05: Check OptiTrack")
        if not self._wait_optitrack_message():
            self._log.warning("  No message — launching optitrack_client and retrying …")
            proc = subprocess.Popen(['ros2', 'run', 'optitrack_client', 'optitrack_client'])
            self._processes['optitrack_client'] = proc
            time.sleep(self._cfg['optitrack']['retry_delay_sec'])
            self._optitrack_event.clear()
            if not self._wait_optitrack_message():
                raise MissionAbortError("OptiTrack did not come up after launching client")
        self._verify_optitrack_header()
        self._log.info("╚══ Stage 05 OK: OptiTrack verified")

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

    def _stage_05b_connect_tello_wifi(self) -> None:
        self._log.info("╔══ Stage 05b: Connect to Tello WiFi")
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
        self._log.info("╚══ Stage 05b OK: Tello WiFi connected")

    def _stage_06_launch_tello_driver(self) -> None:
        self._log.info("╔══ Stage 06: Launch tello_driver")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'tello_driver', 'tello_driver.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['tello_driver'] = proc
        self._log.info(f"  tello_driver launched (pid={proc.pid})")
        delay = float(self._cfg['drone'].get('driver_startup_delay_sec', 15.0))
        self._log.info(f"  Waiting {delay}s for tello_driver to finish configuring …")
        time.sleep(delay)
        self._log.info("╚══ Stage 06 OK")

    def _stage_07_drone_preflight(self) -> None:
        self._log.info("╔══ Stage 07: Drone preflight checks")
        cfg_d = self._cfg['drone']

        # Camera
        self._log.info(f"  Waiting for {cfg_d['camera_topic']} …")
        if not self._camera_event.wait(timeout=cfg_d['camera_timeout_sec']):
            raise MissionAbortError(
                f"Camera topic {cfg_d['camera_topic']} not active after "
                f"{cfg_d['camera_timeout_sec']}s")
        self._log.info(f"  Camera OK: {cfg_d['camera_topic']} is live")

        # Battery
        self._log.info(f"  Waiting for battery ≥ {cfg_d['battery_min_pct']}% …")
        if not self._battery_event.wait(timeout=cfg_d['battery_timeout_sec']):
            pct = self._battery_pct
            pct_str = f"{pct:.1f}%" if pct is not None else "unknown"
            raise MissionAbortError(
                f"Battery check failed: {pct_str} < {cfg_d['battery_min_pct']}%")
        self._log.info(f"  Battery OK: {self._battery_pct:.1f}%")

        # Confirm drone state is -1 (before takeoff)
        if self._drone_state is not None and self._drone_state != -1:
            raise MissionAbortError(
                f"Expected drone state -1 (before takeoff), got {self._drone_state}")
        self._log.info("╚══ Stage 07 OK: Drone is ready for takeoff")

    def _stage_08_launch_tello_map(self) -> None:
        self._log.info("╔══ Stage 08: Launch tello_map (drone take-off + scan)")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'tello_pos_control', 'tello_map.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['tello_map'] = proc
        self._log.info(f"  tello_map launched (pid={proc.pid})")
        self._log.info("╚══ Stage 08 OK")

    def _stage_09_observe_drone_states(self) -> None:
        self._log.info("╔══ Stage 09: Monitor drone state transitions")
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
            if not self._drone_state_events[state].wait(timeout=tmo):
                self._log.error(f"  TIMEOUT in state {state} ({name})")
                self._abort_drone()
                raise MissionAbortError(f"Drone timeout in state {state} ({name})")
            self._log.info(f"  → State {state} ({name}) reached")

        self._log.info("╚══ Stage 09 OK: Drone mission complete (state 4)")

    def _stage_10_wait_video_topics(self) -> None:
        self._log.info("╔══ Stage 10: Wait for video file-path topics")
        cfg_d = self._cfg['drone']
        cfg_v = self._cfg['video']
        timeout = cfg_v['file_appear_timeout_sec']

        self._log.info(
            f"  Waiting for {cfg_d['video_filename_topic']} (timeout={timeout}s) …")
        if not self._video_filename_event.wait(timeout=timeout):
            raise MissionAbortError(
                f"No message on {cfg_d['video_filename_topic']} after {timeout}s")

        self._log.info(
            f"  Waiting for {cfg_d['telemetry_filename_topic']} (timeout={timeout}s) …")
        if not self._telemetry_filename_event.wait(timeout=timeout):
            raise MissionAbortError(
                f"No message on {cfg_d['telemetry_filename_topic']} after {timeout}s")

        self._log.info(f"  video_path:    {self._video_path!r}")
        self._log.info(f"  telemetry_path:{self._telemetry_path!r}")

        for label, path in (('scan.mp4', self._video_path), ('telemetry.csv', self._telemetry_path)):
            if not path or not os.path.isfile(path):
                raise MissionAbortError(f"{label} path does not exist: {path!r}")
            size_kb = os.path.getsize(path) / 1024
            if size_kb == 0:
                raise MissionAbortError(f"{label} is empty: {path!r}")
            self._log.info(f"  {label}: {size_kb:.1f} KB — OK")

        self._log.info("╚══ Stage 10 OK")

    def _stage_11_verify_video_integrity(self) -> None:
        self._log.info("╔══ Stage 11: Verify scan.mp4 integrity via ffmpeg")
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', self._video_path, '-f', 'null', '-'],
            capture_output=True,
        )
        if result.returncode != 0:
            err = result.stderr.decode().strip()
            raise MissionAbortError(
                f"scan.mp4 failed ffmpeg integrity check: {err}")
        self._log.info("╚══ Stage 11 OK: scan.mp4 is valid")

    def _wait_for_publisher(self, topic: str, timeout_sec: float, label: str) -> None:
        """Block until at least one publisher exists on *topic* or raise MissionAbortError."""
        deadline = time.monotonic() + timeout_sec
        while self.count_publishers(topic) == 0:
            if time.monotonic() > deadline:
                raise MissionAbortError(
                    f"{label} not ready: no publisher on '{topic}' after {timeout_sec}s")
            time.sleep(0.5)
        self._log.info(f"  {label} ready — publisher on '{topic}' detected")

    def _stage_12_launch_trajectory_planner(self) -> None:
        self._log.info("╔══ Stage 12: Launch trajectory_planner")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'trajectory_planner', 'trajectory_planner_launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['trajectory_planner'] = proc
        self._log.info(f"  trajectory_planner launched (pid={proc.pid})")
        cfg_tp = self._cfg['trajectory_planner']
        self._wait_for_publisher(
            cfg_tp['ready_topic'], cfg_tp['ready_timeout_sec'], 'trajectory_planner')
        self._log.info("╚══ Stage 12 OK")

    def _stage_13_launch_map_fusion(self) -> None:
        self._log.info("╔══ Stage 13: Launch map_fusion")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'map_fusion', 'map_fusion.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['map_fusion'] = proc
        self._log.info(f"  map_fusion launched (pid={proc.pid})")
        cfg_mf = self._cfg['map_fusion']
        self._wait_for_publisher(
            cfg_mf['ready_topic'], cfg_mf['ready_timeout_sec'], 'map_fusion')
        self._log.info("╚══ Stage 13 OK")

    def _stage_14_launch_oradar(self) -> None:
        self._log.info("╔══ Stage 14: Launch oradar lidar")
        proc = subprocess.Popen(
            ['ros2', 'launch', 'oradar_ros', 'ms200_scan.launch.py'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._processes['oradar'] = proc
        self._log.info(f"  oradar launched (pid={proc.pid})")
        cfg_or = self._cfg['oradar']
        self._wait_for_publisher(
            cfg_or['scan_topic'], cfg_or['ready_timeout_sec'], 'oradar')
        self._log.info("╚══ Stage 14 OK")

    def _stage_15_launch_marker_localizer(self) -> None:
        self._log.info("╔══ Stage 15: Launch arena_marker_localizer")
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
        self._log.info("╚══ Stage 15 OK: marker_localizer ready")

    def _stage_16_call_localize_markers(self) -> List:
        self._log.info("╔══ Stage 16: Call /localize_markers")
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
        self._log.info("╚══ Stage 16 OK")
        return markers

    def _stage_17_publish_aruco_poses(self, markers: List) -> None:
        self._log.info("╔══ Stage 17: Publish /aruco/amr/pose and /aruco/goal/pose")
        cfg_a = self._cfg['aruco']
        amr_id = cfg_a['amr_marker_id']
        goal_id = cfg_a['goal_marker_id']

        by_id = {int(m.id): m for m in markers}

        if amr_id not in by_id:
            raise MissionAbortError(
                f"AMR marker id={amr_id} not found in localizer response "
                f"(found ids: {sorted(by_id.keys())})")
        if goal_id not in by_id:
            raise MissionAbortError(
                f"Goal marker id={goal_id} not found in localizer response "
                f"(found ids: {sorted(by_id.keys())})")

        self._pub_aruco_amr.publish(by_id[amr_id].pose_with_covariance)
        self._pub_aruco_goal.publish(by_id[goal_id].pose_with_covariance)
        self._log.info(f"  Published AMR pose (marker {amr_id}) to {cfg_a['amr_pose_topic']}")
        self._log.info(f"  Published goal pose (marker {goal_id}) to {cfg_a['goal_pose_topic']}")
        self._log.info("╚══ Stage 17 OK")

    def _stage_18_launch_map_builder(self) -> None:
        self._log.info("╔══ Stage 18: Launch arena_map_builder server")
        proc = subprocess.Popen(
            ['ros2', 'run', 'arena_map_builder', 'build_arena_map_server'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
        self._log.info("╚══ Stage 18 OK: map_builder ready")

    def _stage_19_call_map_builder(self) -> OccupancyGrid:
        self._log.info("╔══ Stage 19: Call BuildArenaMap action")
        goal = BuildArenaMap.Goal()
        goal.video_path = self._video_path
        self._log.info(f"  video_path={self._video_path!r}")

        timeout = self._cfg['map_builder']['action_timeout_sec']
        result = self._call_action(goal, timeout_sec=timeout)

        if not result.success:
            raise MissionAbortError(f"BuildArenaMap failed: {result.message}")

        self._log.info(
            f"  Map built: {result.map.info.width}×{result.map.info.height} cells, "
            f"{result.n_obstacles} obstacles, "
            f"mean consistency={result.mean_consistency:.3f}")
        self._log.info("╚══ Stage 19 OK")
        return result.map

    def _stage_20_publish_drone_map(self, grid: OccupancyGrid) -> None:
        self._log.info("╔══ Stage 20: Publish /drone/map")
        cfg_mb = self._cfg['map_builder']

        # Ensure frame_id is 'world' as map_fusion expects
        grid.header.frame_id = 'world'
        grid.header.stamp = self.get_clock().now().to_msg()

        self._pub_drone_map.publish(grid)
        self._log.info(
            f"  Published {grid.info.width}×{grid.info.height} OccupancyGrid "
            f"to {cfg_mb['drone_map_topic']}")

        # Confirm our own subscriber fires (latched; same process)
        if not self._drone_map_event.wait(timeout=5.0):
            self._log.warning("  /drone/map subscriber did not echo back within 5s (may be ok)")
        self._log.info("╚══ Stage 20 OK: /drone/map live")

    # ────────────────────────────────────────────────────────────────────────
    # Main orchestration loop
    # ────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._log.info("━━━━━━━━━━━━━━━━  MISSION START  ━━━━━━━━━━━━━━━━")
        try:
            self._stage_01_ping()
            self._stage_02_ssh_connect()
            self._stage_03_launch_amr()
            self._stage_04_wait_imu_ready()
            self._stage_05_check_optitrack()
            self._stage_05b_connect_tello_wifi()
            self._stage_06_launch_tello_driver()
            self._stage_07_drone_preflight()
            self._stage_08_launch_tello_map()
            self._stage_09_observe_drone_states()
            self._stage_10_wait_video_topics()
            self._stage_11_verify_video_integrity()
            self._stage_12_launch_trajectory_planner()
            self._stage_13_launch_map_fusion()
            self._stage_14_launch_oradar()
            self._stage_15_launch_marker_localizer()
            markers = self._stage_16_call_localize_markers()
            self._stage_17_publish_aruco_poses(markers)
            self._stage_18_launch_map_builder()
            grid = self._stage_19_call_map_builder()
            self._stage_20_publish_drone_map(grid)

            self._mission_complete = True
            self._log.info(
                "━━━━━━  MISSION ORCHESTRATION COMPLETE  ━━━━━━\n"
                "map_fusion and trajectory_planner are now operating autonomously.\n"
                "Press Ctrl+C to exit.")

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
        node._teardown_ssh()  # always stop AMR service on exit
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
