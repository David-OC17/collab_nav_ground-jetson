"""
Integration tests for the collab_nav_ground-jetson security middleware posture.

Each test uses lightweight proxy nodes that mirror the signing level and
min_level requirements of the real stack nodes.  Actual sensor data types are
used so type-matching errors surface early.

Test groups
-----------
A. Signed ↔ signed communication (positive path)
B. Unsigned message dropped by signed subscriber (negative path)
C. Legacy relay vouches for native hardware publishers (relay pattern)
D. Kill switch degrades whole graph to native ROS2
E. Policy-driven min_level enforcement (security_policy.yaml)
"""

import pytest
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from ros2_security import SecureNodeMixin, SecurityLevel
from ros2_security.legacy_relay import LegacyRelayNode


def spin_until(executor, tick_fns, predicate, timeout=5.0):
    """Tick publishers and spin until predicate() is True or timeout elapses."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        for fn in tick_fns:
            fn()
        executor.spin_once(timeout_sec=0.05)
    return predicate()

# ---------------------------------------------------------------------------
# Tiny reusable node helpers
# ---------------------------------------------------------------------------

class _SignedPub(SecureNodeMixin, Node):
    """Signed publisher: publishes at SecurityLevel.SIGN_ONLY."""

    def __init__(self, name, topic, msg_type, certs_dir):
        super().__init__(name)
        self.security_init(level=SecurityLevel.SIGN_ONLY, certs_dir=certs_dir)
        self._pub = self.create_secure_publisher(topic, msg_type, 10)
        self._msg_type = msg_type

    def publish(self, msg=None):
        self.secure_publish(self._pub, msg or self._msg_type())


class _SignedSub(SecureNodeMixin, Node):
    """Signed subscriber: accepts only SecurityLevel.SIGN_ONLY or higher."""

    def __init__(self, name, topic, msg_type, certs_dir, min_level=SecurityLevel.SIGN_ONLY):
        super().__init__(name)
        self.security_init(level=SecurityLevel.SIGN_ONLY, certs_dir=certs_dir)
        self.received = []
        self.create_secure_subscription(
            topic, msg_type, self.received.append,
            min_level=min_level, qos=10,
        )


class _NativePub(Node):
    """Plain, unsigned publisher (simulates uncontrolled C++ driver)."""

    def __init__(self, name, topic, msg_type):
        super().__init__(name)
        self._pub = self.create_publisher(msg_type, topic, 10)
        self._msg_type = msg_type

    def publish(self, msg=None):
        self._pub.publish(msg or self._msg_type())


class _NoneSecurePub(SecureNodeMixin, Node):
    """SecureNodeMixin publisher at SecurityLevel.NONE.

    Publishes native (unsigned) messages through the mixin's publisher path,
    which avoids DDS type-conflict issues while still exercising the
    'unsigned source' scenario that signed subscribers must reject.
    """

    def __init__(self, name, topic, msg_type, certs_dir):
        super().__init__(name)
        self.security_init(level=SecurityLevel.NONE, certs_dir=certs_dir)
        self._pub = self.create_secure_publisher(topic, msg_type, 10)
        self._msg_type = msg_type

    def publish(self, msg=None):
        self.secure_publish(self._pub, msg or self._msg_type())


class _NativeSub(Node):
    """Plain, unsigned subscriber."""

    def __init__(self, name, topic, msg_type):
        super().__init__(name)
        self.received = []
        self.create_subscription(msg_type, topic, self.received.append, 10)


# ---------------------------------------------------------------------------
# A. Signed ↔ signed communication
# ---------------------------------------------------------------------------

def test_signed_pub_signed_sub_odometry(executor, certs_dir):
    """lidar_odometry_node → occupancy_mapper: /amr/odom delivered when both signed."""
    ex, nodes = executor
    pub = _SignedPub('lidar_odometry_node', '/amr/odom', Odometry, certs_dir)
    sub = _SignedSub('occupancy_mapper', '/amr/odom', Odometry, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/amr/odom not delivered between two signed nodes'


def test_signed_pub_signed_sub_occupancy_grid(executor, certs_dir):
    """occupancy_mapper → map_fusion_node: /amr/world_map delivered when both signed."""
    ex, nodes = executor
    pub = _SignedPub('occupancy_mapper', '/amr/world_map', OccupancyGrid, certs_dir)
    sub = _SignedSub('map_fusion_node', '/amr/world_map', OccupancyGrid, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/amr/world_map not delivered between two signed nodes'


def test_signed_pub_signed_sub_emergency_stop(executor, certs_dir):
    """emergency_stop → mission_orchestrator: /amr/emergency_stop delivered."""
    ex, nodes = executor
    pub = _SignedPub('emergency_stop', '/amr/emergency_stop', Bool, certs_dir)
    sub = _SignedSub('mission_orchestrator', '/amr/emergency_stop', Bool, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    stop_msg = Bool(data=True)
    delivered = spin_until(ex, [lambda: pub.publish(stop_msg)], lambda: len(sub.received) > 0)
    assert delivered, '/amr/emergency_stop not delivered between two signed nodes'
    assert sub.received[0].data is True


def test_signed_pub_signed_sub_path(executor, certs_dir):
    """astar_planner2 → spline_follower: /trajectory_planner2/path delivered."""
    ex, nodes = executor
    pub = _SignedPub('astar_planner2', '/trajectory_planner2/path', Path, certs_dir)
    sub = _SignedSub('spline_follower', '/trajectory_planner2/path', Path, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/trajectory_planner2/path not delivered between two signed nodes'


def test_signed_pub_signed_sub_fused_map(executor, certs_dir):
    """map_fusion_node → astar_planner2: /fused_map delivered when both signed."""
    ex, nodes = executor
    pub = _SignedPub('map_fusion_node', '/fused_map', OccupancyGrid, certs_dir)
    sub = _SignedSub('astar_planner2', '/fused_map', OccupancyGrid, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/fused_map not delivered between two signed nodes'


def test_signed_pub_signed_sub_goal(executor, certs_dir):
    """frontier_explorer → astar_planner2: /frontier_explorer/goal delivered."""
    ex, nodes = executor
    pub = _SignedPub('frontier_explorer', '/frontier_explorer/goal',
                     PoseWithCovarianceStamped, certs_dir)
    sub = _SignedSub('astar_planner2', '/frontier_explorer/goal',
                     PoseWithCovarianceStamped, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/frontier_explorer/goal not delivered between two signed nodes'


def test_signed_pub_signed_sub_amr_pose(executor, certs_dir):
    """optitrack_pose_node → downstream: /amr/pose delivered when signed."""
    ex, nodes = executor
    pub = _SignedPub('optitrack_pose_node', '/amr/pose', Odometry, certs_dir)
    sub = _SignedSub('test_signed_sub', '/amr/pose', Odometry, certs_dir)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish], lambda: len(sub.received) > 0)
    assert delivered, '/amr/pose not delivered from signed publisher'


# ---------------------------------------------------------------------------
# B. Unsigned message dropped by signed subscriber
# ---------------------------------------------------------------------------

def test_unsigned_scan_dropped_by_signed_sub(executor, certs_dir):
    """
    A SecurityLevel.NONE source on /scan is dropped by a SIGN_ONLY subscriber.

    Uses _NoneSecurePub (SecureNodeMixin at NONE level) which publishes the
    native LaserScan type — the same type that an uncontrolled C++ oradar
    driver would produce.  The SecureEnvelope subscriber never receives it
    because the NONE publisher produces no signed envelope.
    """
    ex, nodes = executor
    pub = _NoneSecurePub('lidar_odometry_node', '/scan_test_drop', LaserScan, certs_dir)
    sub = _SignedSub('test_signed_sub', '/scan_test_drop', LaserScan, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Unsigned LaserScan must not reach a SIGN_ONLY subscriber'
    assert sub.received == []


def test_unsigned_pose_dropped_by_signed_sub(executor, certs_dir):
    """
    A SecurityLevel.NONE source publishing PoseStamped is dropped by a
    SIGN_ONLY subscriber (mirrors the optitrack_pose_node's subscription
    requirement on /optitrack/rigid_body before the relay is in place).
    """
    ex, nodes = executor
    pub = _NoneSecurePub('optitrack_pose_node', '/optitrack_test_drop', PoseStamped,
                         certs_dir)
    sub = _SignedSub('test_signed_sub', '/optitrack_test_drop', PoseStamped,
                     certs_dir, min_level=SecurityLevel.SIGN_ONLY)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Unsigned PoseStamped must not reach a SIGN_ONLY subscriber'
    assert sub.received == []


def test_unsigned_odom_dropped_by_signed_sub(executor, certs_dir):
    """
    An unsigned (NONE-level) Odometry source is rejected by signed subscribers.
    Ensures rogue Odometry cannot be injected into the navigation pipeline.
    """
    ex, nodes = executor
    pub = _NoneSecurePub('lidar_odometry_node', '/amr_odom_test_drop', Odometry, certs_dir)
    sub = _SignedSub('occupancy_mapper', '/amr_odom_test_drop', Odometry, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Unsigned Odometry must be dropped by SIGN_ONLY subscriber'
    assert sub.received == []


def test_unsigned_bool_dropped_by_signed_sub(executor, certs_dir):
    """
    A rogue NONE-level Bool source cannot inject a false emergency-stop signal
    into a SIGN_ONLY consumer — the mission_orchestrator pattern.
    """
    ex, nodes = executor
    pub = _NoneSecurePub('emergency_stop', '/amr_estop_test_drop', Bool, certs_dir)
    sub = _SignedSub('mission_orchestrator', '/amr_estop_test_drop', Bool, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [lambda: pub.publish(Bool(data=True))],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Unsigned emergency-stop must not reach a SIGN_ONLY subscriber'
    assert sub.received == []


# ---------------------------------------------------------------------------
# C. Legacy relay vouches for native hardware publishers
# ---------------------------------------------------------------------------

def test_relay_vouches_for_scan(executor, certs_dir):
    """
    The scan_relay re-signs raw /scan messages so that secured nodes
    (e.g. lidar_odometry_node) can receive them with min_level=sign.

    Topology: NativePub(/scan_relay_in) → LegacyRelayNode(sign) → SignedSub
    """
    ex, nodes = executor
    native = _NativePub('stub_scan_pub', '/scan_relay_in', LaserScan)
    relay = LegacyRelayNode(
        [(LaserScan, '/scan_relay_in', '/scan_relay_out')],
        level=SecurityLevel.SIGN_ONLY,
        certs_dir=certs_dir,
        node_name='scan_relay',
    )
    sub = _SignedSub('lidar_odometry_node', '/scan_relay_out', LaserScan, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [native, relay, sub]
    for n in nodes:
        ex.add_node(n)

    delivered = spin_until(ex, [native.publish], lambda: len(sub.received) > 0)
    assert delivered, 'Relay-signed LaserScan must be accepted by SIGN_ONLY subscriber'


def test_relay_vouches_for_optitrack_pose(executor, certs_dir):
    """
    The optitrack_relay re-signs raw /optitrack/rigid_body so that
    optitrack_pose_node can accept it.

    Topology: NativePub(/optitrack_raw) → LegacyRelayNode(sign) → SignedSub
    """
    ex, nodes = executor
    native = _NativePub('stub_optitrack_pub', '/optitrack_raw', PoseStamped)
    relay = LegacyRelayNode(
        [(PoseStamped, '/optitrack_raw', '/optitrack_signed')],
        level=SecurityLevel.SIGN_ONLY,
        certs_dir=certs_dir,
        node_name='optitrack_relay',
    )
    sub = _SignedSub('optitrack_pose_node', '/optitrack_signed', PoseStamped, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [native, relay, sub]
    for n in nodes:
        ex.add_node(n)

    delivered = spin_until(ex, [native.publish], lambda: len(sub.received) > 0)
    assert delivered, 'Relay-signed PoseStamped must be accepted by SIGN_ONLY subscriber'


def test_relay_without_cert_unsigned_still_dropped(executor, certs_dir):
    """
    A relay that publishes at level=NONE (no signing) cannot satisfy a
    SIGN_ONLY subscriber even though a relay is present.  Only a relay
    operating at sign level with a valid cert is accepted.
    """
    ex, nodes = executor
    # NONE-level relay: subscribes natively, publishes natively (no signing)
    relay = LegacyRelayNode(
        [(LaserScan, '/scan_none_in', '/scan_none_out')],
        level=SecurityLevel.NONE,
    )
    native = _NativePub('stub_scan_pub', '/scan_none_in', LaserScan)
    sub = _SignedSub('test_signed_sub', '/scan_none_out', LaserScan, certs_dir,
                     min_level=SecurityLevel.SIGN_ONLY)
    nodes += [relay, native, sub]
    for n in nodes:
        ex.add_node(n)

    delivered = spin_until(ex, [native.publish],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Unsigned relay output must still be dropped by SIGN_ONLY subscriber'


# ---------------------------------------------------------------------------
# D. Kill switch degrades whole graph to native ROS2
# ---------------------------------------------------------------------------

def test_kill_switch_allows_native_through(executor, certs_dir, monkeypatch):
    """
    With ROS2_SECURITY_DISABLED=1 all security checks are bypassed: both
    sides degrade to native ROS2 so any publisher's messages reach any subscriber.

    The kill switch is module-level (evaluated at import time).  We patch the
    constant in all three modules that reference it so the change is consistent.
    """
    import ros2_security.security_manager as _sm
    import ros2_security.secure_node_mixin as _mixin

    monkeypatch.setattr(_sm, 'SECURITY_ENABLED', False)
    monkeypatch.setattr(_mixin, 'SECURITY_ENABLED', False, raising=False)

    ex, nodes = executor

    class _KillPub(SecureNodeMixin, Node):
        def __init__(self):
            super().__init__('test_signed_pub')
            self.security_init(level=SecurityLevel.SIGN_ONLY, certs_dir=certs_dir)
            self._pub = self.create_secure_publisher('/kill_switch_test', Bool, 10)

        def publish(self, msg=None):
            self.secure_publish(self._pub, msg or Bool())

    class _KillSub(SecureNodeMixin, Node):
        def __init__(self):
            super().__init__('test_signed_sub')
            self.security_init(level=SecurityLevel.SIGN_ONLY, certs_dir=certs_dir)
            self.received = []
            self.create_secure_subscription(
                '/kill_switch_test', Bool, self.received.append,
                min_level=SecurityLevel.SIGN_ONLY, qos=10,
            )

    pub = _KillPub()
    sub = _KillSub()
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish],
                           lambda: len(sub.received) > 0)
    assert delivered, 'Kill switch should pass messages between mixin nodes degraded to native'


# ---------------------------------------------------------------------------
# E. Policy-driven min_level enforcement
# ---------------------------------------------------------------------------

def test_policy_scan_min_level_sign_drops_unsigned(executor, certs_dir, tmp_path):
    """
    security_policy.yaml declares min_level=sign for /scan on lidar_odometry_node.
    An unsigned publisher must be dropped even when no explicit min_level arg is
    passed to create_secure_subscription.
    """
    policy = tmp_path / 'security_policy.yaml'
    policy.write_text(
        'global_min_level: none\n'
        'nodes:\n'
        '  lidar_odometry_node:\n'
        '    publish_level: sign\n'
        '    subscriptions:\n'
        '      /scan_policy_test:\n'
        '        min_level: sign\n'
    )

    ex, nodes = executor

    class _PolicyNode(SecureNodeMixin, Node):
        def __init__(self):
            super().__init__('lidar_odometry_node')
            self.security_init(certs_dir=certs_dir, policy_path=str(policy))
            self.received = []
            # No explicit min_level: resolved from policy (sign).
            self.create_secure_subscription('/scan_policy_test', LaserScan, self.received.append)

    # NONE-level source: publishes native LaserScan (no envelope, no signature).
    # The policy subscriber will see no SecureEnvelope messages → drop.
    pub = _NoneSecurePub('scan_relay', '/scan_policy_test', LaserScan, certs_dir)
    sub = _PolicyNode()
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    delivered = spin_until(ex, [pub.publish],
                           lambda: len(sub.received) > 0, timeout=2.0)
    assert not delivered, 'Policy-driven min_level=sign must drop unsigned /scan'


def test_policy_emergency_stop_min_level_sign_accepted(executor, certs_dir, tmp_path):
    """
    security_policy.yaml requires sign on /amr/emergency_stop.
    A signed publisher (emergency_stop node) satisfies the policy requirement.
    """
    policy = tmp_path / 'security_policy.yaml'
    policy.write_text(
        'global_min_level: none\n'
        'nodes:\n'
        '  mission_orchestrator:\n'
        '    publish_level: sign\n'
        '    subscriptions:\n'
        '      /amr/emergency_stop:\n'
        '        min_level: sign\n'
    )

    ex, nodes = executor
    pub = _SignedPub('emergency_stop', '/amr/emergency_stop', Bool, certs_dir)

    class _PolicyOrchestrator(SecureNodeMixin, Node):
        def __init__(self):
            super().__init__('mission_orchestrator')
            self.security_init(certs_dir=certs_dir, policy_path=str(policy))
            self.received = []
            self.create_secure_subscription(
                '/amr/emergency_stop', Bool, self.received.append)

    sub = _PolicyOrchestrator()
    nodes += [pub, sub]
    ex.add_node(pub)
    ex.add_node(sub)

    stop_msg = Bool(data=True)
    delivered = spin_until(ex, [lambda: pub.publish(stop_msg)],
                           lambda: len(sub.received) > 0)
    assert delivered, 'Policy-compliant signed /amr/emergency_stop must be delivered'
    assert sub.received[0].data is True
