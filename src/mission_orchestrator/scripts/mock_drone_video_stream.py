#!/usr/bin/env python3
"""
Mock the drone pipeline to test the arena_map_builder map-creation flow
(online OR offline) WITHOUT flying the drone or running the rest of the mission.

Like save_scan_data.py, this reuses the real MissionOrchestratorNode and simply
overrides / no-ops the components we don't want to run. Here we:

  • no-op the drone bring-up (01-05), AMR bring-up (07-10), the planner / fusion /
    lidar / e-stop (11-14b) and the marker-localizer / aruco / AMR-mapper stages
    (15, 16, 17, 17b, 17c) — none of which the map builder needs;
  • mock the drone itself by replaying a PRE-SAVED scan.mp4 (which already holds
    the undistorted+flipped+cropped frames the controller would publish) onto the
    processed-image topic at a fixed rate during "stage 05" — emulating the drone
    streaming over ROS;
  • run the MAP-BUILDER pipeline exactly as the real mission does: the real
    _stage_18 brings up the server, and online_start / online_stop / _stage_19 /
    _stage_20 run unchanged.

Mode is taken from the config (map_builder.online), so the SAME script tests
both:
  • online  : the server is brought up in online mode before "stage 05"; the
    mock streams frames which the server stitches live; online_stop finalizes and
    the BuildArenaMap action runs only transfer + occupancy.
  • offline : the server is brought up after "stage 16" and stitches the saved
    scan.mp4 in the action; no live stream is needed (none is sent).

Usage (from workspace root, after sourcing install/setup.bash):
    # uses whatever map_builder.online says in the config
    python3 src/mission_orchestrator/scripts/mock_drone_video_stream.py --scan-id 20

    # force a mode regardless of config
    python3 src/mission_orchestrator/scripts/mock_drone_video_stream.py --scan-id 20 --mode online
    python3 src/mission_orchestrator/scripts/mock_drone_video_stream.py --scan-id 20 --mode offline

    # relay an arbitrary video file
    python3 src/mission_orchestrator/scripts/mock_drone_video_stream.py --video /abs/scan.mp4 --mode online
"""

from __future__ import annotations

import argparse
import array
import os
import sys
import threading
import time

import cv2

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, '..', '..', '..'))

sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPTS_DIR, '..')))

from mission_orchestrator.orchestrator_node import (  # noqa: E402
    MissionOrchestratorNode,
    MissionAbortError,
    _kill_proc,
)
from mission_orchestrator.scan_data_io import scan_video_dir  # noqa: E402

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(_SCRIPTS_DIR, '..', 'config', 'orchestrator_params.yaml'))


def _bgr_to_imgmsg(node, frame, frame_id: str = 'camera') -> Image:
    """Build a bgr8 sensor_msgs/Image from a contiguous BGR ndarray.

    NOTE: assign the uint8[] `data` field as array.array('B', ...), NOT raw
    bytes. rclpy's setter takes a ~150 ms/msg slow path for a `bytes` value
    (element-wise), capping the rate near 7 fps; the array.array path is a fast
    C-level copy (~0.2 ms/msg). This is what lets the relay hit 30 fps.
    """
    h, w = frame.shape[:2]
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = array.array('B', frame.tobytes())
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator subclass: real map-builder stages, mocked drone + everything else
# ─────────────────────────────────────────────────────────────────────────────

class _MockDroneOrchestrator(MissionOrchestratorNode):
    """Runs the real map-builder pipeline (online/offline per config) while a
    pre-saved video is replayed onto the processed-image topic as the drone."""

    _relay_video_path: str = ''
    _relay_fps: float = 30.0

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self._relay_pub = None   # created lazily so cfg overrides apply

    # ── drone mock ────────────────────────────────────────────────────────────

    def _ensure_relay_pub(self):
        if self._relay_pub is None:
            topic = self._cfg['map_builder'].get(
                'online_image_topic', '/camera/image_proc')
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=2)
            self._relay_pub = self.create_publisher(Image, topic, qos)
            self._relay_topic = topic

    def _relay_video(self) -> None:
        """Replay the video onto the processed-image topic at _relay_fps,
        emulating the drone controller publishing live frames."""
        self._ensure_relay_pub()
        cap = cv2.VideoCapture(self._relay_video_path)
        if not cap.isOpened():
            raise MissionAbortError(f"cannot open video: {self._relay_video_path!r}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        period = 1.0 / max(self._relay_fps, 1e-3)
        self._log.info(
            f"  streaming {self._relay_video_path!r} → {self._relay_topic} "
            f"@ {self._relay_fps:.0f} fps (source {src_fps:.0f} fps, {n_total} frames)")

        sent = 0
        t_next = time.monotonic()
        try:
            while rclpy.ok():
                ret, frame = cap.read()
                if not ret:
                    break
                self._relay_pub.publish(_bgr_to_imgmsg(self, frame))
                sent += 1
                if sent % 60 == 0:
                    pct = 100.0 * sent / max(n_total, 1)
                    self._log.info(f"    … streamed {sent}/{n_total} frames ({pct:.0f}%)")
                t_next += period
                sleep = t_next - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    t_next = time.monotonic()   # fell behind; don't accrue debt
        finally:
            cap.release()
        self._log.info(f"  streamed {sent} frames total")

    # ── Stage 03.f override: the drone "flight" = replay the video ────────────

    def _stage_03f_observe_drone_states(self) -> None:
        if self._online_enabled:
            self._log.info("╔══ Stage 03.f (MOCK): streaming video → emulating drone flight")
            self._relay_video()
            self._log.info("╚══ Stage 03.f (MOCK) OK: stream complete (drone 'landed')")
        else:
            self._log.info(
                "╔══ Stage 03.f (MOCK): offline mode — no live stream; the server "
                "will stitch the saved video in the map build")
            self._log.info("╚══ Stage 03.f (MOCK) OK")

    # ── Stage 03.g override: point the pipeline at the saved video, no waiting ─

    def _stage_03g_wait_video_files(self) -> None:
        self._video_path = self._relay_video_path
        self._telemetry_path = os.path.join(
            os.path.dirname(self._relay_video_path), 'telemetry.csv')
        self._log.info(f"╠══ Stage 03.g (MOCK): video_path={self._video_path!r}")

    # ── Stage 04.b override: no localizer here → just join the map build ──────

    def _stage_04b_publish_aruco_poses(self, markers) -> None:
        result = self._await_map_result()
        self._publish_drone_map(result.map)
        gp, ap_ = result.goal_marker_position, result.amr_marker_position
        self._log.info(
            f"╠══ Stage 04.b (MOCK): map markers (m)  "
            f"goal=({gp.x:.3f}, {gp.y:.3f})  amr=({ap_.x:.3f}, {ap_.y:.3f})")

    # 04.c no-op: the map is already published above; skip the quality
    # classifier and the frontier-exploration fallback for this mock stream.
    def _stage_04c_classify_and_branch(self, markers) -> None: pass

    # ── No-ops: optitrack, drone bring-up, localizer, vslam, rasp, mapping ───

    def _stage_01a_check_optitrack(self) -> None: pass
    def _stage_01b_optitrack_sanity(self) -> None: pass
    def _stage_03a_connect_tello_wifi(self) -> None: pass
    def _stage_03b_launch_tello_driver(self) -> None: pass
    def _stage_03c_drone_preflight(self) -> None: pass
    def _stage_03d_launch_tello_map(self) -> None: pass
    def _stage_03h_verify_video_integrity(self) -> None: pass
    def _stage_04_launch_marker_localizer(self) -> None: pass
    def _stage_04a_call_localize_markers(self):
        return []
    def _stage_05a_verify_realsense(self) -> None: pass
    def _stage_05b_start_vslam(self) -> None: pass
    def _stage_05c_check_vslam_odometry(self) -> None: pass
    def _stage_06a_ping(self) -> None: pass
    def _stage_06b_ssh_connect(self) -> None: pass
    def _stage_06c_launch_amr(self) -> None: pass
    def _stage_06d_wait_imu_ready(self) -> None: pass
    def _stage_07a_emergency_stop(self) -> None: pass
    def _stage_08a_launch_oradar(self) -> None: pass
    def _stage_08b_publish_static_tf(self) -> None: pass
    def _stage_08c_amr_mapper(self) -> None: pass
    def _stage_09_trajectory_planner(self) -> None: pass
    def _stage_10_observer(self) -> None: pass

    # Don't run the real drone-abort sequence (publishes /land + sleeps).
    def _abort_drone(self) -> None: pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test the arena_map_builder map-creation pipeline (online or "
                    "offline) by mocking the drone: replay a pre-saved video onto "
                    "the processed-image topic while the real map-builder stages run.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--scan-id', type=int, metavar='N',
                     help='Replay scan.mp4 from src/arena_map_builder/data/drone_scans/scanN/')
    src.add_argument('--video', metavar='PATH', help='Path to a video file to replay')
    ap.add_argument('--config', default=_DEFAULT_CONFIG,
                    help='orchestrator_params.yaml (default: config/ inside this package)')
    ap.add_argument('--mode', choices=['online', 'offline'], default=None,
                    help='Override map_builder.online from the config')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='Replay rate in frames/sec (default 30)')
    ap.add_argument('--topic', default=None,
                    help='Override map_builder.online_image_topic (the processed-image topic)')
    ap.add_argument('--background', default=None,
                    help='Override map_builder.background_image_path')
    args = ap.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"ERROR: config not found: {args.config}")

    if args.video:
        video_path = args.video
    else:
        video_path = os.path.join(scan_video_dir(_WORKSPACE_ROOT, args.scan_id), 'scan.mp4')
    if not os.path.isfile(video_path):
        sys.exit(f"ERROR: video not found: {video_path}")

    rclpy.init()
    node = _MockDroneOrchestrator(args.config)

    # Apply overrides (must happen before run(); the relay pub + stage 18 read cfg).
    mb = node._cfg.setdefault('map_builder', {})
    if args.mode is not None:
        mb['online'] = (args.mode == 'online')
        node._online_enabled = (args.mode == 'online')
    if args.topic:
        mb['online_image_topic'] = args.topic
    if args.background:
        mb['background_image_path'] = args.background
    node._relay_video_path = video_path
    node._relay_fps = args.fps

    node._log.info(
        f"[mock] mode={'ONLINE' if node._online_enabled else 'OFFLINE'}  "
        f"video={video_path!r}  fps={args.fps:.0f}  "
        f"topic={mb.get('online_image_topic', '/camera/image_proc')}")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True, name='ros-spin')
    spin_thread.start()

    try:
        node.run()
        if node._mission_complete:
            node._log.info("══════ MOCK MAP-BUILD TEST COMPLETE ══════")
    except KeyboardInterrupt:
        pass
    finally:
        # Tear down anything the real stages launched (the map-builder server).
        for name, proc in list(node._processes.items()):
            _kill_proc(proc, name, node._log)
        node._stop_rosbag()
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
