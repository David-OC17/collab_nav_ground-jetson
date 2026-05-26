"""
Integration tests for the post-scan pipeline: stages 11-20.

Stages 01-10 are always no-ops; the mocker injects scan.mp4 + telemetry.csv
from the scan10 dataset (the same source DroneMockNode uses) to simulate the
drone having already landed and published its files.

Two test variants:

  Mocked  — stages 12-14 (trajectory_planner, map_fusion, oradar) are replaced
             with FakeProc + a lightweight ROS 2 publisher that satisfies the
             readiness check (_wait_for_publisher).  Stages 15-20 run for real.
             Use this for fast CI feedback without needing the full nav stack.

  Integration — only stages 01-10 are no-ops; stages 11-20 all run for real,
                including launching trajectory_planner, map_fusion and oradar_ros.
                Requires the full workspace to be built and hardware available
                (oradar LiDAR connected).  Mark with @pytest.mark.integration and
                run explicitly:  pytest -m integration

The integration test is skipped by default unless -m integration is passed.
"""

from __future__ import annotations

import os
import shutil

import pytest
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import String

from test.mocks.drone_mock import SCAN10_DIR
from test.test_helpers import TestableOrchestratorNode, _FakeProc, make_test_config


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_WORKSPACE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'install'))
_BACKGROUND_IMAGE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 '..', '..', 'arena_map_builder', 'config', 'background.png'))
_SCAN_MP4 = os.path.join(SCAN10_DIR, 'scan.mp4')
_TELEMETRY_CSV = os.path.join(SCAN10_DIR, 'telemetry.csv')


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight publisher mock nodes (satisfy _wait_for_publisher in mocked test)
# ─────────────────────────────────────────────────────────────────────────────

class _TrajectoryPlannerMock(Node):
    """Advertises /trajectory_planner/path to satisfy the stage 12 readiness check."""

    def __init__(self) -> None:
        super().__init__('trajectory_planner_mock')
        self._pub = self.create_publisher(Path, '/trajectory_planner/path', 10)


class _MapFusionMock(Node):
    """Advertises /fusion/status to satisfy the stage 13 readiness check."""

    def __init__(self) -> None:
        super().__init__('map_fusion_mock')
        self._pub = self.create_publisher(String, '/fusion/status', 10)


class _OradarMock(Node):
    """Advertises /scan to satisfy the stage 14 readiness check."""

    def __init__(self) -> None:
        super().__init__('oradar_mock')
        from sensor_msgs.msg import LaserScan
        self._pub = self.create_publisher(LaserScan, '/scan', 10)


# ─────────────────────────────────────────────────────────────────────────────
# Shared orchestrator base: stages 01-10 no-ops, video paths injected
# ─────────────────────────────────────────────────────────────────────────────

class _PipelineOrchestratorBase(TestableOrchestratorNode):
    """
    Stages 01-10 are no-ops.  Stage 10 injects the scan10 video and telemetry
    paths directly, simulating what the drone would have published.
    """

    def _stage_01_ping(self): pass
    def _stage_02_ssh_connect(self): pass
    def _stage_03_launch_amr(self): pass
    def _stage_04_wait_imu_ready(self): pass
    def _stage_05_check_optitrack(self): pass
    def _stage_05b_connect_tello_wifi(self): pass
    def _stage_06_launch_tello_driver(self): pass
    def _stage_07_drone_preflight(self): pass
    def _stage_08_launch_tello_map(self): pass
    def _stage_09_observe_drone_states(self): pass

    def _stage_10_wait_video_topics(self):
        self._log.info("╔══ Stage 10: [MOCK] injecting scan10 video + telemetry")
        self._video_path = _SCAN_MP4
        self._telemetry_path = _TELEMETRY_CSV
        self._log.info(f"  video:     {self._video_path}")
        self._log.info(f"  telemetry: {self._telemetry_path}")
        self._log.info("╚══ Stage 10 OK (mocked)")


# ─────────────────────────────────────────────────────────────────────────────
# Mocked variant: stages 12-14 use FakeProc + publisher mocks
# ─────────────────────────────────────────────────────────────────────────────

class _PipelineMockedOrchestrator(_PipelineOrchestratorBase):
    """Stages 11-14 mocked; stages 15-20 are real."""

    def _stage_11_verify_video_integrity(self):
        self._log.info("╔══ Stage 11: [MOCK] skipping ffmpeg check")
        self._log.info("╚══ Stage 11 OK (mocked)")

    def _stage_12_launch_trajectory_planner(self):
        self._log.info("╔══ Stage 12: [MOCK] trajectory_planner")
        self._processes['trajectory_planner'] = _FakeProc()
        self._log.info("╚══ Stage 12 OK (mocked)")

    def _stage_13_launch_map_fusion(self):
        self._log.info("╔══ Stage 13: [MOCK] map_fusion")
        self._processes['map_fusion'] = _FakeProc()
        self._log.info("╚══ Stage 13 OK (mocked)")

    def _stage_14_launch_oradar(self):
        self._log.info("╔══ Stage 14: [MOCK] oradar")
        self._processes['oradar'] = _FakeProc()
        self._log.info("╚══ Stage 14 OK (mocked)")


# ─────────────────────────────────────────────────────────────────────────────
# Config builder for pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_config(tmp_path) -> str:
    return make_test_config(tmp_path, overrides={
        'marker_localizer': {
            'workspace_path': _WORKSPACE_PATH,
            'service_timeout_sec': 120.0,
            'server_ready_timeout_sec': 30.0,
        },
        'map_builder': {
            'background_image_path': _BACKGROUND_IMAGE,
            'action_timeout_sec': 600.0,
            'server_ready_timeout_sec': 30.0,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Run helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(executor, node, *extra_nodes):
    for n in extra_nodes:
        executor.add_node(n)
    executor.add_node(node)
    try:
        node.run()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        for n in extra_nodes:
            executor.remove_node(n)
            n.destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# Mocked test (fast, no hardware)
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_mocked(executor, tmp_path):
    """
    Stages 01-14 mocked; stages 15-20 run for real against the scan10 dataset.

    Requires: arena_marker_localizer and arena_map_builder built in the workspace.
    """
    cfg = _pipeline_config(tmp_path)

    tp_mock = _TrajectoryPlannerMock()
    mf_mock = _MapFusionMock()
    or_mock = _OradarMock()
    node = _PipelineMockedOrchestrator(cfg)

    _run(executor, node, tp_mock, mf_mock, or_mock)

    assert node.mission_complete, "Pipeline stages 11-20 should all complete"
    assert not node.abort_called


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (real nodes, requires full workspace + hardware)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')
def test_pipeline_integration(executor, tmp_path):
    """
    Stages 01-10 mocked; stages 11-20 all run for real including launching
    trajectory_planner, map_fusion, and oradar_ros.

    Requires:
      - Full workspace built (trajectory_planner, map_fusion, oradar_ros,
        arena_marker_localizer, arena_map_builder)
      - Oradar MS200 LiDAR connected (for /scan readiness check)
    """
    cfg = _pipeline_config(tmp_path)
    node = _PipelineOrchestratorBase(cfg)

    _run(executor, node)

    assert node.mission_complete, "Full pipeline stages 11-20 should all complete"
    assert not node.abort_called
