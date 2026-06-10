# Mission Orchestrator
## Technical Documentation

---

## 1. Overview

The mission orchestrator is the top-level supervisory component of the collab\_nav system. It is responsible for the ordered initialization of all subsystems, the supervised execution of the aerial scan and map-building pipeline, and the handoff of control to the autonomous navigation stack once all preconditions for ground navigation have been satisfied.

The orchestrator executes as a single ROS 2 node running on the Jetson Orin Nano. It is the sole entry point for bringing the full system from a cold start to an autonomously navigating AMR. After handoff, the orchestrator transitions to a passive observer role, logging live telemetry from the navigation stack without intervening in its operation.

---

## 2. State Machine Architecture

### 2.1 Formal Model

The orchestrator is modeled as a **deterministic linear finite-state machine**:

$$M = (Q,\ \Sigma,\ \delta,\ q_0,\ F)$$

where:

- $Q = \{S_1, S_2, \ldots, S_{10},\ \texttt{ABORT},\ \texttt{COMPLETE}\}$ — the set of states, one per top-level stage plus two terminal states.
- $\Sigma = \{\texttt{success},\ \texttt{abort}\}$ — the input alphabet.
- $\delta$ — the transition function:

$$
\delta(S_i,\ \texttt{success}) = \begin{cases} S_{i+1} & i \in \{1,\ldots,9\} \\ \texttt{COMPLETE} & i = 10 \end{cases}
\qquad
\delta(S_i,\ \texttt{abort}) = \texttt{ABORT} \quad \forall\, i
$$

- $q_0 = S_1$ — initial state.
- $F = \{\texttt{COMPLETE}\}$ — the single accepting state.

Each state $S_i$ is internally decomposed into an ordered sequence of **substages** $S_{i.a},\ S_{i.b}, \ldots$, forming an inner FSM with the same transition semantics: any substage failure propagates the `abort` signal to the outer machine.

### 2.2 State Diagram

```
          ┌────────────────────────────────────────────────────────────────┐
          │              abort (MissionAbortError from any S_i)            │
          │                                                                 │
          ▼                                                                 │
      ┌───────┐                                                             │
      │ ABORT │  ◄──────────────────────────────────────────────────────── ┤
      └───────┘                                                             │
                                                                            │
 q₀   success                                                               │
  │                                                                         │
  ▼                                                                         │
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                          │
│  S₁  │─►│  S₂  │─►│  S₃  │─►│  S₄  │─►│  S₅  │─────────────────────────┤
│OTrk  │  │Map   │  │Drone │  │ArUco │  │VSLAM │                          │
│Brup  │  │Brup  │  │Scan  │  │Loc.  │  │Brup  │                          │
└──────┘  └──────┘  └──────┘  └──────┘  └──────┘                          │
                                                                            │
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                          │
│  S₁₀ │◄─│  S₉  │◄─│  S₈  │◄─│  S₇  │◄─│  S₆  │◄────────────────────────┘
│Obs.  │  │Plan  │  │Mapp  │  │EStop │  │AMR   │
│Mode  │  │Brup  │  │Brup  │  │Vrfy  │  │Brup  │
└──────┘  └──────┘  └──────┘  └──────┘  └──────┘
  │
  ▼
┌──────────┐
│ COMPLETE │
└──────────┘
```

### 2.3 Within-State Transition Model

Most substages require waiting for an external condition to become true (a topic to publish, a service to become available, a file to appear). The general polling model used throughout is a **fixed-interval deadline poll**:

```
function poll_until(condition, T_deadline, T_poll):
    t₀ ← now()
    loop:
        if condition():
            return success
        if now() − t₀ ≥ T_deadline:
            raise MissionAbortError
        sleep(T_poll)
```

All `T_deadline` values are configurable parameters. `T_poll` is fixed per substage (values in the range 1–5 s). There is no exponential backoff: failed substages are retried at a constant interval until the deadline is exceeded, at which point the mission aborts.

For substages that invoke external commands (WiFi scan, parameter set) a **fixed-count retry** model is used instead:

```
function retry_n(action, n_max):
    for attempt in 1 .. n_max:
        if action() succeeds:
            return success
        sleep(T_sleep)
    raise MissionAbortError
```

---

## 3. Stage Decomposition

### Stage 01 — OptiTrack Bringup

The OptiTrack motion-capture system serves as the absolute positioning reference for the UAV throughout the scan. This stage verifies that the VRPN-to-ROS bridge is live and that the incoming pose stream is geometrically valid.

| Substage | Description | Abort condition |
|---|---|---|
| 01.a | Check for an active publisher on the OptiTrack pose topic. If none is found, the VRPN client node is launched automatically and the check is retried once. | No message within timeout after client launch |
| 01.b | Sanity-check the incoming message: verify that the `frame_id` matches the configured expected value, and that the message timestamp is within a configurable freshness window $\Delta t_{\max}$. | Frame ID mismatch; stamp age $> \Delta t_{\max}$ |

The freshness check enforces $t_{\text{now}} - t_{\text{stamp}} \leq \Delta t_{\max}$, guarding against a scenario in which the VRPN bridge is publishing stale data buffered from a prior session.

### Stage 02 — Aerial Map Builder Bringup

This stage brings up the stitching server and configures it before the drone flies. Configuration must precede flight because in online mode (see §10) the server begins consuming the drone's camera stream immediately at takeoff.

| Substage | Description | Abort condition |
|---|---|---|
| 02.a | Launch the map-builder server process. Wait for its ROS 2 action server to become available. Set the background image path via a runtime parameter using a fixed-count retry (up to 5 attempts, 1 s between attempts) to account for node-discovery latency. | Server not available within timeout; parameter set fails after 5 attempts |
| 02.b | Log and confirm the configured stitching mode (online or offline). No external action. | — |

### Stage 03 — UAV Scan Routine

This stage manages the complete DJI Tello flight lifecycle: connectivity, preflight safety checks, scan execution, and artifact verification.

| Substage | Description | Abort condition |
|---|---|---|
| 03.a | Scan for the Tello's Wi-Fi access point using `nmcli`. Up to `scan_retries` scans are attempted with 2 s between them. On success, the Jetson connects to the Tello network. | SSID not visible after all retries; connection fails |
| 03.b | Launch the Tello ROS 2 driver process. A fixed startup delay is observed to allow the driver to complete hardware handshake. | — |
| 03.c | Preflight safety checks: verify the drone's camera topic is publishing (live sensor feed), read the battery state and abort if below the configured minimum percentage, and confirm the drone's internal state is pre-takeoff. | Camera topic timeout; battery below minimum; unexpected state |
| 03.d | Launch the position-controlled scan routine. In online stitching mode, the controller is instructed to republish processed image frames to the map-builder's intake topic. | — |
| 03.e | *(Online mode only)* Call the map-builder's `online_start` service, signaling it to begin consuming the live frame stream. | Service call fails |
| 03.f | Monitor the drone's state machine via the `/drone/state` topic. The expected sequence is state 1 (stabilize) → 2 (trajectory) → 3 (return) → 4 (landing), each with an independent timeout. The drone is commanded to land and the driver is stopped if any timeout is exceeded. | Timeout on any state transition |
| 03.g | Wait for the scan video and telemetry CSV to appear on disk, checking for non-zero file size and that the modification time is within a configurable age window (the file must be fresh, not a leftover from a prior run). | Files absent, empty, or stale within timeout |
| 03.h | Verify the integrity of the scan video by running an ffmpeg null-output decode pass. A non-zero return code indicates a corrupt or truncated file. | ffmpeg error |

The drone state machine observed in substage 03.f is published by the position controller (`tello_pos_control`). The expected integer values and their semantics are:

| Value | Meaning |
|---|---|
| −1 | Pre-takeoff (initial state, checked in 03.c) |
| 1 | Stabilize (hover at scan altitude) |
| 2 | Executing lawnmower scan trajectory |
| 3 | Returning to home position |
| 4 | Landing sequence |

### Stage 04 — Marker Localization

Stage 04 is the only stage that exploits **structured concurrency**: the map build and marker localization run as overlapping tasks, joined at substage 04.b. The rationale and structure of this overlap are described in §5.

| Substage | Description | Abort condition |
|---|---|---|
| (kickoff) | After 03.h passes: finalize the live stitch (online mode) and dispatch the `BuildArenaMap` action asynchronously. The result future is held and joined at 04.b. | Action rejected |
| 04 | Launch the marker localizer service node and wait for the service to become available. | Service not available within timeout |
| 04.a | Call `/localize_markers` with the scan video and telemetry CSV. Returns the **orientation** (PnP-derived) of all detected ArUco markers in the map frame. This call executes while the map build proceeds in the background. | Service returns failure; required marker IDs absent |
| 04.b | Join the background map-build result. Extract the **position** (pixel-projected) of the goal and AMR markers from the map builder's result. Fuse orientation (from 04.a) and position (from map builder) into `PoseWithCovarianceStamped` messages and publish them on latched topics. Also publish the `OccupancyGrid` to the drone map topic. | Map action timeout or failure; map builder returns NaN position for a required marker |
| 04.c | Pass the 45-element map diagnostic feature vector (returned by the `BuildArenaMap` action) to the map quality classifier (§6). If the map is classified as acceptable, the regular pipeline continues. If not, frontier exploration is activated as a fallback while the pipeline continues. | — (no abort; fallback is activated instead) |

The position–orientation fusion at 04.b is motivated by the complementary accuracy of the two sources: the map builder's pixel-projection provides metric coordinates anchored to the physical arena scale, while the PnP solver provides sub-degree angular accuracy from detected ArUco corner correspondences. The two are expressed in the same reference frame (the OccupancyGrid origin), so only the position component of the localizer's output is replaced.

### Stage 05 — Visual SLAM Bringup

The Isaac ROS Visual SLAM (cuSLAM) pipeline provides Visual-Inertial Odometry (VIO) as one of the three sources fused by the EKF on the RDK X3. Its initialization is independent of the map-building pipeline and can therefore proceed in parallel with stages 01–04 in principle; in the current linear execution model it is placed here.

| Substage | Description | Abort condition |
|---|---|---|
| 05.a | Verify that the Intel RealSense D435i depth camera is enumerated on the USB bus. | Camera not detected |
| 05.b | Execute the VSLAM startup script, which starts the Isaac ROS Docker container and launches the cuSLAM ROS 2 nodes inside it. | Script returns non-zero exit code |
| 05.c | Wait for a valid odometry message on the VSLAM output topic, confirming that the tracking pipeline has initialized and is producing estimates. | Topic timeout |

### Stage 06 — AMR Platform Bringup

The AMR's drive platform (Yahboom RDK X3) runs as a separate computer reachable over Wi-Fi. This stage establishes connectivity and starts the AMR's ROS 2 bringup service remotely.

| Substage | Description | Abort condition |
|---|---|---|
| 06.a | Ping the RDK X3 at its configured static IP. | Host unreachable |
| 06.b | Open an SSH connection to the RDK X3 using the Paramiko library. The connection is kept open through all subsequent SSH interactions (stages 06.c, teardown). | Authentication or connection timeout |
| 06.c | Issue `systemctl start <amr_service>` over SSH. Poll `systemctl is-active` until the service reaches the `active` state or the deadline is exceeded. On timeout, the systemd journal tail is captured and logged for diagnosis. | Service does not reach `active` state within timeout |
| 06.d | Wait for the configured minimum number of IMU messages to arrive on the IMU topic, confirming that the RDK X3's sensor pipeline is publishing. This is a proxy for EKF readiness: the EKF node requires a stable IMU stream to initialize. | IMU message count not reached within timeout |

### Stage 07 — Emergency Stop Verification

The Automatic Emergency Braking (AEB) node monitors the LiDAR point cloud and publishes a Boolean signal on a dedicated topic. Before the AMR is allowed to move, this stage verifies that the AEB system is healthy and not reporting an obstruction.

| Substage | Description | Abort condition |
|---|---|---|
| 07.a | Launch the emergency stop node. Collect a configurable number of consecutive messages from the AEB output topic. If any message reports an active stop condition, or if the required number of messages is not received within the deadline, the mission aborts. | Active stop condition detected; topic timeout |

The message-count check (rather than a single-message check) guards against a spurious initial transient that may occur during node startup before the LiDAR scan is stable.

### Stage 08 — Mapping Bringup

This stage brings up the components responsible for the AMR's local world representation: the LiDAR driver, the coordinate frame alignment node, and the occupancy mapper.

| Substage | Description | Abort condition |
|---|---|---|
| 08.a | Launch the OraDAR MS200 LiDAR driver. Wait for a publisher to appear on the scan topic. | Scan topic timeout |
| 08.b | Launch the coordinate alignment node, which reads the latched `/aruco/amr/pose` (published in stage 04.b) and broadcasts the static `world → odom` TF transform. This transform anchors the AMR's odometry frame to the global map frame established by the aerial survey. | — |
| 08.c | Launch the odometry-based occupancy mapper. | — |

The `world → odom` transform published in 08.b is the critical link between the aerial map and the AMR's local sensor data. Without it, the LiDAR occupancy grid would be expressed in the odometry frame, which drifts over time; with it, the mapper accumulates scan data in the same coordinate frame as the drone-derived `OccupancyGrid`.

### Stage 09 — Trajectory Planner Bringup

The trajectory planner is launched last, after the map and coordinate frame are in place, because it immediately begins consuming the occupancy grid and the goal pose to plan a path.

| Substage | Description | Abort condition |
|---|---|---|
| 09 | Launch the trajectory planner. Wait for a publisher to appear on the planner's readiness topic, confirming that the A* server is active and has received a valid goal pose from the latched `/aruco/goal/pose` topic. | Readiness topic timeout |

### Stage 10 — Observer Mode

Upon successful completion of stage 09, the orchestrator declares the mission handoff complete. The autonomous navigation stack (mapper + trajectory planner + AEB) now operates independently. The orchestrator enters **observer mode**: it subscribes to a configurable set of telemetry topics and logs formatted snapshots to the structured log at a configurable rate. It does not publish, command, or interfere with the navigation stack.

The observer pattern provides operational visibility — position, velocity, IMU, and pose estimates all in a single log line — without coupling the orchestrator to the control loop.

---

## 4. Bringup Dependency Graph

The ordering of stages is not arbitrary: each stage satisfies preconditions required by subsequent stages. The dependency structure is as follows:

```
OptiTrack (S₁)
    └──► required by: drone position controller (S₃)

Map Builder configured (S₂)
    └──► required by: online frame intake (S₃.d/e)

Drone scan complete (S₃) + Map Builder action running (async)
    └──► required by: localization inputs (S₄)

ArUco poses + OccupancyGrid published (S₄)
    ├──► /aruco/amr/pose  → world→odom TF (S₈.b)
    ├──► /aruco/goal/pose → trajectory planner goal (S₉)
    └──► /drone/map       → map fusion input (S₈.c, S₉)

VSLAM odometry live (S₅)
    └──► required by: EKF fusion on RDK X3 (running after S₆)

AMR bringup + IMU live (S₆)
    └──► required by: AEB scan available (S₇), mapping (S₈), planning (S₉)

AEB verified inactive (S₇)
    └──► safety precondition for any motion command

world→odom TF + LiDAR live (S₈)
    └──► required by: trajectory planner map input (S₉)
```

The only deliberate deviation from a strictly sequential dependency order is the **S₄ async overlap** described in §5: the map build and localization are independent computations that share only their inputs, so they are pipelined.

---

## 5. Concurrent Map Build and Localization (Stage 04)

The `BuildArenaMap` action (full stitching pipeline) and the `/localize_markers` service call (PnP-based orientation estimation) are both computationally expensive and share the same inputs (scan video and telemetry CSV). They are independent in computation and can therefore be executed concurrently. The orchestrator exploits this with the following structure:

```
after stage 03 completes:

  ┌─ Background thread ─────────────────────────────────────────────┐
  │  (online mode) call online_stop → finalize live stitch           │
  │  send BuildArenaMap(video_path) → result_future                 │
  │  [map stitching, obstacle extraction, occupancy encoding ...]    │
  └─────────────────────────────────────────────────────────────────┘

  ┌─ Main thread (stage 04.a) ──────────────────────────────────────┐
  │  launch arena_marker_localizer                                   │
  │  call /localize_markers(video_path, telemetry_csv)               │
  │  → orientation_markers  (PnP-derived, runs while map builds)    │
  └─────────────────────────────────────────────────────────────────┘

  stage 04.b:
    position_data ← await result_future   // join background task
    for each required marker m:
        pose_m.position    ← position_data.marker_position[m]  // metric, from map
        pose_m.orientation ← orientation_markers[m].orientation // angular, from PnP
        pose_m.covariance  ← orientation_markers[m].covariance
    publish pose_m on latched topic
    publish OccupancyGrid to /drone/map
```

In the worst case (where localization completes before the map build) the main thread blocks at the `await` call until the background action finishes. In practice, the PnP call is faster than the full stitch, so this overlap typically reduces total wall-clock latency compared to sequential execution.

---

## 6. Map Quality Decision and Fallback (Stage 04.c)

The `BuildArenaMap` action returns, alongside the `OccupancyGrid`, a **45-element diagnostic feature vector** characterizing the quality of the stitched map (e.g., obstacle blob count, green-hull convexity, inter-frame alignment residuals, coverage fraction). The orchestrator feeds this vector to a trained **Random Forest classifier** to decide whether the map is reliable enough to serve as the primary navigation input.

The decision rule is:

$$
\text{map acceptable} \iff P(\text{pass} \mid \mathbf{f}) \geq \tau
$$

where $\mathbf{f} \in \mathbb{R}^{45}$ is the feature vector, $P(\text{pass} \mid \mathbf{f})$ is the class-conditional probability estimated by the forest, and $\tau$ is a threshold determined during training to balance precision and recall on the held-out validation set.

The model is exported as a portable NumPy array (no scikit-learn dependency at inference time), making it deployable on the Jetson without a full ML environment. Missing features — which can occur if an upstream diagnostic step failed — are filled with a sentinel value of $-1.0$, which the training data associates with degraded maps; this ensures the classifier fails safe in the presence of incomplete data.

**Decision outcomes:**

| $P(\text{pass} \mid \mathbf{f}) \geq \tau$ | Action |
|---|---|
| True | Regular pipeline continues (map fusion → A* planning) |
| False | Frontier exploration activated as parallel fallback; regular pipeline also continues |

The fallback activation is non-aborting: the mission does not stop, but the AMR is additionally instructed to explore frontiers in case the goal marker cannot be localized from the degraded map. The two strategies run concurrently and whichever locates the goal first determines the navigation target.

---

## 7. Fault Handling and Abort Sequence

Any substage raises a `MissionAbortError` exception upon detecting an unrecoverable condition. This exception propagates through the call stack to the top-level `run()` method, which catches it and initiates the **abort sequence**.

The abort sequence follows two governing principles:

1. **Drone safety has priority.** A flying drone whose software controller has been killed is a physical hazard. The abort sequence therefore stops the scan controller, issues a land command through the driver (which remains alive specifically for this purpose), and waits a configurable duration for the landing to complete before terminating the driver.

2. **Graceful shutdown before forced termination.** All managed subprocesses receive `SIGINT` first, allowing ROS 2 nodes to execute their cleanup callbacks (undeclare parameters, publish terminal messages, etc.). If a process does not exit within a configurable timeout, `SIGKILL` is sent.

The abort sequence proceeds as follows:

```
abort():
    1. Drone abort (if drone was active):
       a. Kill scan controller (SIGINT → timeout → SIGKILL)
       b. Publish /land command via subprocess
       c. Wait T_land_wait seconds for physical landing
       d. Kill Tello driver (SIGINT → timeout → SIGKILL)

    2. Stop rosbag recording (if active)

    3. For each remaining managed subprocess:
       send SIGINT → wait T_sigint_timeout → send SIGKILL if still alive

    4. SSH teardown:
       stop AMR systemd service on RDK X3
       close SSH connection
```

The abort sequence is **idempotent**: calling it twice has no additional effect, which is important because both `KeyboardInterrupt` and `MissionAbortError` paths reach the same handler.

---

## 8. Inter-Computer Communication Architecture

The system spans three computing nodes (Jetson Orin Nano, Yahboom RDK X3, DJI Tello) connected over two physical networks. The communication architecture is partitioned into a **control plane** and a **data plane**:

```
                    ┌─────────────────────────────────────────────────────┐
                    │               Jetson Orin Nano                      │
                    │           (mission_orchestrator)                    │
                    └───────────┬─────────────────┬───────────────────────┘
                                │                 │
              SSH / TCP         │                 │  Tello SDK / UDP
           (control plane)      │                 │  (2.4 GHz Wi-Fi AP)
                                │                 │
                    ┌───────────▼────────┐    ┌───▼──────────────────────┐
                    │  Yahboom RDK X3    │    │     DJI Tello UAV        │
                    │  (AMR drive + EKF) │    │  (onboard controller)    │
                    └───────────┬────────┘    └──────────────────────────┘
                                │
                    ROS 2 DDS / Wi-Fi
                    (data plane, shared domain)
                                │
                    ┌───────────▼────────────────────────────────────────┐
                    │              Shared ROS 2 Network                   │
                    │  /amr/ekf/odom · /imu/data_raw · /scan · etc.      │
                    └────────────────────────────────────────────────────┘

  OptiTrack Server ──VRPN/ETH──► optitrack_client ──ROS 2──► Jetson
```

**Control plane (SSH):** The orchestrator uses SSH (Paramiko) for lifecycle management of the RDK X3. Remote operations are limited to starting and stopping the AMR systemd service and polling its status. A persistent SSH session is opened in stage 06.b and torn down in the abort/shutdown sequence. This design keeps the Jetson in control of the AMR's software lifecycle without requiring a ROS 2 service dedicated to lifecycle management.

**Data plane (ROS 2 DDS):** All runtime sensor data and navigation messages flow over ROS 2 DDS on the shared Wi-Fi network. The two computers share a `ROS_DOMAIN_ID`, making all topics and services transparently discoverable across machines. Latched QoS is used for map and pose topics to guarantee that late-joining subscribers (e.g., the trajectory planner launched after the map is published) receive the most recent value immediately upon subscription.

**UAV channel (Tello SDK over UDP):** The Tello does not run ROS 2 onboard. The `tello_driver` node on the Jetson bridges the proprietary Tello UDP SDK to ROS 2 topics, translating velocity commands and state reports. The Jetson connects to the Tello's own Wi-Fi access point for this link, which is a separate network from the robot LAN.

**Localization channel (VRPN over Ethernet):** The OptiTrack motion-capture server publishes rigid-body poses via the VRPN protocol over Ethernet. The `optitrack_client` node subscribes to VRPN and republishes poses as ROS 2 `PoseStamped` messages, making OptiTrack data available on the shared DDS network.

---

## 9. Online vs. Offline Stitching Mode

The map builder supports two operating modes, selected at orchestrator startup via a configuration flag:

- **Online mode (default):** The stitching server subscribes to the drone's processed image stream and incrementally builds the composite image during the flight. After the drone lands, stitching is finalized and the `BuildArenaMap` action transfers and encodes the result. This minimizes the post-landing latency before a usable map is available.

- **Offline mode:** The stitching server receives no frames during the flight. After landing, the full pipeline (stitch + transfer + encode) is triggered by the `BuildArenaMap` action against the saved scan video. This mode is used when the network bandwidth between the drone and the map-builder server is insufficient for real-time frame delivery.

The orchestrator's stage 03 is the only point where the two modes diverge significantly (online start/stop service calls). The rest of the pipeline is identical. Full details of the stitching pipeline are documented in the `arena_map_builder` DOCS.md.

---

## 10. Testing

A set of hardware test scripts (`scripts/run_hw_test_*.py`) covers subsets of the full stage sequence, enabling staged integration testing without executing the complete mission. These scripts exercise individual subsystem pairs (drone only, post-scan AMR navigation, drone-to-AMR handoff) using recorded scan data, decoupling hardware availability constraints during development.

A unit test suite (`test/`) exercises the orchestrator's stage logic against in-process mock ROS 2 nodes, covering happy-path and failure-path scenarios for OptiTrack, drone state transitions, EKF stability, and SSH connectivity, without requiring physical hardware.
