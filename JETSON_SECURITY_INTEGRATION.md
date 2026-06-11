# Jetson ↔ Raspberry Pi — Secure Integration Guide

How to bring the **Jetson Nano** into the `ros2_security` graph so it talks to the
Raspberry Pi seamlessly. The Pi side is already migrated (see
[`config/security_policy.yaml`](config/security_policy.yaml) and the
`secure_*_launch.py` files). This document is the **contract** the Jetson must
honour — get these eight things right and the two machines interoperate with no
further changes.

> TL;DR: same CA, same policy file, same `SecureEnvelope` message build, same
> kill-switch state, synced clocks, same `ROS_DOMAIN_ID`/RMW. Then sign on the
> topics the Pi requires signed, and leave the third-party VSLAM behind a NONE
> legacy relay.

---

## 0. The system is **one** security domain, split across two machines

There is a single CA, a single auditable `security_policy.yaml`, and a single
`SecureEnvelope` wire type. The Pi and Jetson are just two hosts inside it. Every
rule below exists to keep those three things **identical** on both hosts.

Anything that diverges silently breaks discovery or drops traffic:

| If this differs between hosts | Symptom |
| --- | --- |
| Kill-switch state (`ROS2_SECURITY_DISABLED`) | DDS type mismatch (`SecureEnvelope` vs native) → nodes never connect |
| `SecureEnvelope.msg` / pkg version | DDS type-hash mismatch → never connect |
| CA (`ca.crt`) | Signature verify fails → every signed message dropped |
| System clock (> 30 s skew) | Replay guard drops every signed message |
| `ROS_DOMAIN_ID` / RMW implementation | No cross-host discovery at all |

---

## 1. Build the identical security packages on the Jetson

The Jetson must build the **same commit** of the security submodule as the Pi, so
the `SecureEnvelope` type hash matches byte-for-byte.

```bash
# On the Jetson, inside its colcon workspace:
sudo apt install python3-cryptography python3-yaml openssl

# Use the SAME submodule commit as the Pi (check: git -C security_middleware rev-parse HEAD)
source /opt/ros/humble/setup.bash
colcon build --packages-select ros2_security_msgs
source install/setup.bash
colcon build --packages-select ros2_security
source install/setup.bash
```

Pin the submodule to the same SHA on both machines. If the Pi later bumps the
submodule, rebuild on the Jetson too.

---

## 2. One CA, distributed — never two

The root of trust is `certs/ca.crt`. **Keep `ca.key` on exactly one machine** (the
Pi is fine, or better, an offline box) and mint every certificate there. Do
**not** run `generate_certs.sh` independently on the Jetson — that would create a
second, untrusted CA.

Mint the Jetson node certs on the machine holding `ca.key` (the existing CA is
reused, certs are added incrementally):

```bash
# On the Pi (which already has certs/ca.key):
CERTS_DIR=./certs ./security_middleware/scripts/generate_certs.sh \
    visual_slam_relay amr_reference jetson_planner
#   ^ add a CN for every Jetson node that will publish/subscribe at a non-NONE level
```

### What to copy to the Jetson

Into the Jetson's `certs/` directory put:

- `ca.crt` — the shared CA (public).
- **its own** `<node>.key` + `<node>.crt` for each Jetson node.
- the **public** `<node>.crt` of every **Pi** node whose signatures the Jetson
  must verify (e.g. `ekf_node.crt` if the Jetson subscribes to `/amr/ekf/odom`).

Symmetrically, copy each **Jetson** node's public `<node>.crt` into the **Pi's**
`certs/` so the Pi can verify the Jetson's signatures.

> A node trusts a remote sender iff `<sender_cn>.crt` is present in its local
> `certs/` **and** verifies against `ca.crt`. Private `.key` files never leave
> the host that owns them. The simplest correct policy: **every host holds all
> public `.crt`s + `ca.crt`, and only its own `.key`s.**

`certs/` is gitignored on both repos — distribute out-of-band (scp/ansible), not
via git.

---

## 3. Share the **same** `security_policy.yaml`

The policy is system-wide, not per-host. The Jetson reads the *same* file content
the Pi does (point `ROS2_SECURITY_POLICY` at a local copy that is kept in sync).
**Add the Jetson's nodes to this one policy** — a node with no entry falls back to
`global_min_level: none` and will silently publish unsigned.

Pi-side topics the Jetson must satisfy are already defined; see the contract in
§5. When you add a Jetson publisher, give it a `publish_level` and list the
subscriptions it makes.

---

## 4. Match runtime environment on every launched process

Set these the **same** on both hosts (use `SetEnvironmentVariable` in launch files
so children inherit them before importing `ros2_security`):

```bash
export ROS_DOMAIN_ID=<same on both>          # e.g. 0
export RMW_IMPLEMENTATION=<same on both>      # e.g. rmw_fastrtps_cpp
export ROS2_SECURITY_POLICY=/abs/path/to/security_policy.yaml
# Kill switch: MUST be identical on both hosts. Leave unset (or =0) in production.
# export ROS2_SECURITY_DISABLED=1            # only if BOTH hosts set it
```

**Clock sync is mandatory.** The replay guard rejects any signed message whose
`ts` is more than 30 s from the receiver's `time.time()`. Run NTP/chrony on both
hosts (or sync the Jetson to the Pi). Without it, cross-host signed traffic dies
intermittently as clocks drift.

> Need a looser bound? `security_init(replay_window=...)` is per node, but change
> it consistently — it does not relax the need for *roughly* synced clocks.

---

## 5. Topic contract (the interop ABI)

Exact topic names, types, and required levels. The Jetson must publish/subscribe
**secured** (via `create_secure_publisher` / `create_secure_subscription` or a
legacy relay) on these — native pub/sub on a secured topic is a type mismatch.

### Jetson → Pi

| Topic | Type | Level the Pi requires | How the Jetson should provide it |
| --- | --- | --- | --- |
| `/visual_slam/tracking/odometry` | `nav_msgs/Odometry` | consumed at **none** (via relay) | **Leave native.** Isaac VSLAM publishes plainly; a `legacy_relay` (NONE) bridges it to `/visual_slam/tracking/odometry_secure`. See §6. |
| `/amr/reference` | `nav_msgs/Odometry` | **sign** | Jetson planner must **sign** it (secured node with a CA cert), or vouch via a `--level sign` relay. See §7. |

### Pi → Jetson (only if the Jetson consumes Pi state)

| Topic | Type | Level (signed by) | Jetson side |
| --- | --- | --- | --- |
| `/amr/ekf/odom` | `nav_msgs/Odometry` | sign (`ekf_node`) | Secured subscription, `min_level: sign`; needs `ekf_node.crt`. |
| `/amr/ekf/vio_accepted` | `nav_msgs/Odometry` | sign (`ekf_node`) | Same as above (debug/telemetry). |
| `/amr/vel_raw`, `/amr/imu/data_raw` | `geometry_msgs/Twist`, `sensor_msgs/Imu` | sign (`driver_node`) | Secured subscription, `min_level: sign`. |

`/tf` (`world`→`base_footprint`) stays **native** by design — the middleware does
not wrap TF. The OptiTrack/`world` provider and the EKF's `odom`→`base_footprint`
broadcast are unchanged.

---

## 6. The VSLAM (`/visual_slam/tracking/odometry`) — keep the NONE relay

Isaac VSLAM is third-party/unmodifiable, so it stays a native, unsigned source and
is bridged at **NONE**. The relay currently runs **on the Pi**; equivalently it
can run on the Jetson (closer to the source). Either way, run it as a process —
do not subclass a node:

```bash
ros2 run ros2_security legacy_relay \
    --bridge nav_msgs/msg/Odometry \
    /visual_slam/tracking/odometry \
    /visual_slam/tracking/odometry_secure
```

The Pi EKF subscribes to `/visual_slam/tracking/odometry_secure` at
`min_level: none` (already in the policy). **Run the relay on exactly one host**,
not both, or you double-publish. Recommended: run it on the Jetson so only the
secured envelope crosses the network.

> If you later want the VSLAM *vouched* at `sign` (so the EKF can require sign on
> VIO), give the relay a `visual_slam_relay` cert and add `--level sign
> --certs-dir ./certs`; then raise the EKF's `/visual_slam/tracking/odometry_secure`
> `min_level` to `sign` in the policy. Coordinate that change on both hosts.

---

## 7. The reference trajectory (`/amr/reference`) — must be signed

The controller requires `min_level: sign` on `/amr/reference` (it drives motion —
an unsigned reference must not be accepted). Pick one:

- **Preferred — sign it natively on the Jetson.** Write the planner as a
  `SecureNodeMixin` node, `publish_level: sign`, CN added to the policy, with a
  CA-signed cert on the Jetson. Mirror the Pi pattern (see
  `src/amr_bringup/amr_bringup/reference_node.py`).
- **Planner unmodifiable — vouch via relay.** Run an inbound relay at `--level
  sign` with a `<relay_cn>` cert that re-signs the planner's native output:

  ```bash
  ros2 run ros2_security legacy_relay --level sign --certs-dir ./certs \
      --bridge nav_msgs/msg/Odometry /amr/reference_raw /amr/reference
  ```

Do **not** lower `/amr/reference` to `min_level: none` — that re-opens the motion
path to spoofing.

> Note: the Pi repo ships an `amr_reference` node. In production the planner lives
> on the Jetson; decide who owns `/amr/reference` and run only **one** publisher.
> If the Jetson owns it, use CN `amr_reference` (reuse the existing policy entry)
> or add a new CN and a matching entry — but not both publishers at once.

---

## 8. Bring-up checklist

1. [ ] Same security-submodule SHA built on both hosts (`ros2_security_msgs` then
       `ros2_security`).
2. [ ] One CA. `ca.key` on a single host; `ca.crt` + needed public `.crt`s on
       both; each host holds only its own `.key`s.
3. [ ] Identical `security_policy.yaml` on both; every Jetson node has an entry.
4. [ ] `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, and kill-switch state identical on
       both hosts.
5. [ ] NTP/chrony running; clock skew < 30 s.
6. [ ] VSLAM relay running on exactly one host (recommend Jetson).
7. [ ] `/amr/reference` published **signed** by exactly one publisher.
8. [ ] Smoke test below passes.

### Smoke test (no robot motion)

On the Jetson, after sourcing, confirm it can verify a Pi signature and that the
relay path works. Echo a secured Pi topic with the trusted cert set:

```bash
# Verifies + decodes a signed Pi topic on the Jetson (proves CA + clock + DDS all agree):
ros2 run ros2_security secure_echo \
    --topic /amr/ekf/odom --type nav_msgs/msg/Odometry \
    --certs-dir ./certs --min-level sign
```

If that prints decoded odometry, the cross-host trust chain is good. A
`[SECURITY] Dropped` warning instead means CA mismatch, missing `ekf_node.crt`,
clock skew, or a kill-switch/`SecureEnvelope`-version divergence — walk §1–§5.

The Pi side has a hardware-free equivalent (`secure_sim_launch.py` +
`tests/test_security_integration.py`) you can mirror on the Jetson to validate its
own nodes before connecting the two.

---

## Quick reference — security levels in play

| Topic | Level | Rationale |
| --- | --- | --- |
| `/amr/cmd_vel`, `/amr/emergency_stop` | **sign** | Motor/safety commands — never accept unsigned. |
| `/amr/reference` | **sign** | Drives motion. |
| `/amr/vel_raw`, `/amr/imu/data_raw`, `/amr/ekf/*` | **sign** | Sensor/state integrity. |
| `/visual_slam/tracking/odometry` → `_secure` | **none** (relayed) | Third-party VSLAM; gated to a velocity-only EKF update. |
| `/tf` | native | Not wrapped by the middleware. |
