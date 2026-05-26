"""
Integration tests for mission_orchestrator stages 06-09 (Tello drone pipeline).

The _OrchestratorUnderTest subclass:
 - no-ops stages 01-05 (covered by test_rasp / test_optitrack)
 - overrides stage 06 to inject a _FakeProc (no real tello_driver launch)
 - overrides stage 08 to inject a _FakeProc AND trigger the DroneMockNode
   state machine (replaces the real tello_map launch)
 - no-ops stages 10-20

The DroneMockNode publishes camera/battery from creation, and starts the
drone state machine only after start_mission() is called (from stage 08).
"""

from __future__ import annotations

from test.test_helpers import TestableOrchestratorNode, _FakeProc, make_test_config
from test.mocks.drone_mock import DroneMockNode


# ─────────────────────────────────────────────────────────────────────────────
# Restricted orchestrator: only runs stages 06-09
# ─────────────────────────────────────────────────────────────────────────────

class _OrchestratorUnderTest(TestableOrchestratorNode):
    """
    Stages 01-05 and 10-20 are no-ops.
    Stage 06 injects a _FakeProc; stage 08 injects a _FakeProc and triggers
    the drone mock state machine.
    """

    def __init__(self, config_path: str, drone_mock: DroneMockNode) -> None:
        super().__init__(config_path)
        self._drone_mock = drone_mock

    def _stage_01_ping(self):
        pass

    def _stage_02_ssh_connect(self):
        pass

    def _stage_03_launch_amr(self):
        pass

    def _stage_04_wait_imu_ready(self):
        pass

    def _stage_05_check_optitrack(self):
        pass

    def _stage_05b_connect_tello_wifi(self):
        pass

    def _stage_06_launch_tello_driver(self) -> None:
        self._log.info("╔══ Stage 06: [MOCK] tello_driver")
        self._processes['tello_driver'] = _FakeProc()
        self._log.info("╚══ Stage 06 OK (mocked)")

    # _stage_07_drone_preflight is inherited — drone mock publishes camera + battery

    def _stage_08_launch_tello_map(self) -> None:
        self._log.info("╔══ Stage 08: [MOCK] tello_map → drone.start_mission()")
        self._processes['tello_map'] = _FakeProc()
        self._drone_mock.start_mission()
        self._log.info("╚══ Stage 08 OK (mocked)")

    # _stage_09_observe_drone_states is inherited — waits for states 1→2→3→4

    def _stage_10_wait_video_topics(self):
        pass

    def _stage_11_verify_video_integrity(self):
        pass

    def _stage_12_launch_trajectory_planner(self):
        pass

    def _stage_13_launch_map_fusion(self):
        pass

    def _stage_14_launch_oradar(self):
        pass

    def _stage_15_launch_marker_localizer(self):
        pass

    def _stage_16_call_localize_markers(self):
        return []

    def _stage_17_publish_aruco_poses(self, markers):
        pass

    def _stage_18_launch_map_builder(self):
        pass

    def _stage_19_call_map_builder(self):
        return None

    def _stage_20_publish_drone_map(self, grid):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build node pair, run, teardown
# ─────────────────────────────────────────────────────────────────────────────

def _run_test(executor, cfg_path, drone_mock: DroneMockNode) -> _OrchestratorUnderTest:
    """Add both nodes to *executor*, run the orchestrator, then clean up."""
    node = _OrchestratorUnderTest(cfg_path, drone_mock)
    executor.add_node(node)
    executor.add_node(drone_mock)
    try:
        node.run()
    finally:
        executor.remove_node(node)
        executor.remove_node(drone_mock)
        node.destroy_node()
        drone_mock.destroy_node()
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_drone_happy_path(executor, tmp_path):
    """Stages 06-09 all succeed: camera/battery OK, states 1→2→3→4 reached."""
    cfg = make_test_config(tmp_path)
    drone = DroneMockNode(video_dir=str(tmp_path / 'video'))

    node = _run_test(executor, cfg, drone)

    assert node.mission_complete, "Mission should complete when all drone states are reached"
    assert not node.abort_called


def test_drone_abort_battery_low(executor, tmp_path):
    """Stage 07 fails: battery below minimum → battery_event never sets → timeout."""
    cfg = make_test_config(tmp_path, overrides={
        'drone': {'battery_min_pct': 50.0, 'battery_timeout_sec': 2.0}
    })
    drone = DroneMockNode(video_dir=str(tmp_path / 'video'), battery_pct=10.0)

    node = _run_test(executor, cfg, drone)

    assert node.abort_called, "Abort should be triggered when battery is too low"
    assert not node.mission_complete


def test_drone_abort_camera_timeout(executor, tmp_path):
    """Stage 07 fails: camera topic never published → camera_event timeout → abort."""
    cfg = make_test_config(tmp_path, overrides={
        'drone': {'camera_timeout_sec': 2.0}
    })
    drone = DroneMockNode(
        video_dir=str(tmp_path / 'video'),
        camera_active=False,
    )

    node = _run_test(executor, cfg, drone)

    assert node.abort_called, "Abort should be triggered when camera is not active"
    assert not node.mission_complete


def test_drone_abort_state1_timeout(executor, tmp_path):
    """Stage 09 fails: drone publishes state 0 then freezes; state 1 never arrives."""
    cfg = make_test_config(tmp_path, overrides={
        'drone': {'state1_timeout_sec': 2.0}
    })
    drone = DroneMockNode(
        video_dir=str(tmp_path / 'video'),
        stuck_at_state=0,
    )

    node = _run_test(executor, cfg, drone)

    assert node.abort_called, "Abort should be triggered when state 1 is never reached"
    assert not node.mission_complete


def test_drone_abort_state2_timeout(executor, tmp_path):
    """Stage 09 fails: drone reaches state 1 but freezes; state 2 never arrives."""
    cfg = make_test_config(tmp_path, overrides={
        'drone': {'state2_timeout_sec': 2.0}
    })
    drone = DroneMockNode(
        video_dir=str(tmp_path / 'video'),
        stuck_at_state=1,
    )

    node = _run_test(executor, cfg, drone)

    assert node.abort_called, "Abort should be triggered when state 2 is never reached"
    assert not node.mission_complete


def test_drone_abort_state3_timeout(executor, tmp_path):
    """Stage 09 fails: drone reaches state 2 but freezes; state 3 never arrives."""
    cfg = make_test_config(tmp_path, overrides={
        'drone': {'state3_timeout_sec': 2.0}
    })
    drone = DroneMockNode(
        video_dir=str(tmp_path / 'video'),
        stuck_at_state=2,
    )

    node = _run_test(executor, cfg, drone)

    assert node.abort_called, "Abort should be triggered when state 3 is never reached"
    assert not node.mission_complete
