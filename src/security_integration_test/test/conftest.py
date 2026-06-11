"""
pytest fixtures for security integration tests.

``ros_context``     — initialises rclpy once for the whole session.
``certs_dir``       — generates a full set of certificates mirroring the
                      production node names used in this project.
``spin``            — helper that ticks publishers and spins until a predicate
                      is satisfied or a timeout expires.
"""

import os
import subprocess
import time

import pytest

# Path to the upstream generate_certs.sh inside the security_middleware submodule.
_HERE = os.path.dirname(os.path.abspath(__file__))
# test/ -> security_integration_test/ -> src/ -> project_root/
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_GENERATE_CERTS = os.path.join(_PROJ_ROOT, 'security_middleware', 'scripts', 'generate_certs.sh')

# All node names used in the collab_nav_ground-jetson production stack.
_STACK_NODES = [
    # Legacy relay bridges (uncontrolled C++ drivers)
    'optitrack_relay',
    'scan_relay',
    # Controlled Python nodes
    'optitrack_pose_node',
    'aruco_localizer',
    'emergency_stop',
    'lidar_odometry_node',
    'occupancy_mapper',
    'map_fusion_node',
    'astar_planner2',
    'spline_follower',
    'frontier_explorer',
    'explorer_controller',
    'mission_orchestrator',
    # Test-only helper nodes
    'test_signed_pub',
    'test_signed_sub',
    'test_relay_node',
]


@pytest.fixture(scope='session')
def ros_context():
    import rclpy
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='session')
def certs_dir(tmp_path_factory):
    """Mint a temporary CA + per-node certs for all nodes in the stack."""
    certs = tmp_path_factory.mktemp('certs')
    env = dict(os.environ, CERTS_DIR=str(certs))
    subprocess.run(
        ['bash', _GENERATE_CERTS, *_STACK_NODES],
        check=True,
        env=env,
        capture_output=True,
    )
    return str(certs)


@pytest.fixture()
def executor(ros_context):
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    managed = []
    yield ex, managed
    for n in managed:
        try:
            ex.remove_node(n)
            n.destroy_node()
        except Exception:
            pass


def spin_until(executor, tick_fns, predicate, timeout=5.0):
    """Tick publishers and spin until predicate() is True or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        for fn in tick_fns:
            fn()
        executor.spin_once(timeout_sec=0.05)
    return predicate()
