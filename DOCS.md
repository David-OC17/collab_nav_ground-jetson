# Drone-Guided Autonomous Navigation: A Collaborative UAV–AMR System
## Technical Documentation

**Authors:** Ortiz, David · Romo, Alfredo · Rosales, Noé · Pulido, Sebastian · Gonzales, Jorge  
**Institution:** Tecnológico de Monterrey, Campus Guadalajara  
**Partners:** OMRON Automation · Nuclea Solutions  
**Platform:** ROS 2 Humble · Yahboom RDK X3 + Jetson Orin Nano · DJI Tello

---

## 1. Problem Statement

Maritime ports operate in GPS-denied or GPS-unreliable environments at ground level. Moving cargo autonomously requires knowing: (a) where the robot is, (b) where it must go, and (c) what obstacles lie in between — all without external absolute positioning infrastructure available at floor level.

The core challenge is therefore **GPS-denied cooperative localization and navigation** in a partially unknown environment, where:

- The **AMR** (Autonomous Mobile Robot) cannot independently determine the global position of the goal zone.
- The **UAV** (Unmanned Aerial Vehicle) has access to absolute positioning at altitude (via OptiTrack, acting as surrogate GPS) but cannot carry cargo.
- Neither robot alone can produce a map accurate enough for reliable path planning.

The solution is a cooperative pipeline: the UAV surveys the arena from above to build a global map and locate goal markers, then hands off structured information to the AMR, which fuses it with its own onboard sensing to plan and execute navigation.

**Test arena:** 4 × 4 m course with cardboard-box obstacles and ArUco markers placed on the floor. No pre-built map is provided at runtime.

---

## 2. Hardware Platform

### 2.1 Ground Robot (AMR)

| Component | Details |
|---|---|
| Base platform | Yahboom RDK X3 (quad-core ARM Cortex-A53, ROS 2 Humble) |
| Co-processor | NVIDIA Jetson Orin Nano (8 GB) |
| LiDAR | OraDAR MS200 (2D, 360°) |
| Onboard IMU | Integrated in RDK X3 |
| Wheel encoders | Hall-effect quadrature encoders on drive wheels |
| Camera | USB wide-angle camera (forward-facing, for ArUco detection) |

The RDK X3 handles low-level motor control, EKF odometry, and hardware I/O. The Jetson Orin Nano handles heavy compute: image stitching, map fusion, trajectory planning, and mission orchestration.

### 2.2 UAV

| Component | Details |
|---|---|
| Platform | DJI Tello (720p downward + forward cameras) |
| Localization | OptiTrack motion-capture system (sub-mm, acts as GPS substitute) |
| Communication | Wi-Fi 2.4 GHz (proprietary SDK over UDP) |

### 2.3 Infrastructure

| Component | Role |
|---|---|
| OptiTrack server | Provides absolute 6-DoF pose of the Tello and AMR via VRPN over Ethernet |
| Wi-Fi AP | Isolates robot LAN from campus internet |
| ROS host PC | Bridges OptiTrack ETH → ROS topics, runs some heavy nodes |

Network topology: `OptiTrack ──ETH── ROS host ──ETH── AP ──WiFi── Jetson / RDK X3 / Tello`

---

## 3. Software Architecture

The system runs **ROS 2 Humble** across two on-robot computers (RDK X3 and Jetson Orin Nano) and a ground-station host. Communication between nodes uses standard DDS (Fast-RTPS), with an optional application-layer **SecureEnvelope** middleware (`network_bridge` submodule) wrapping sensitive topics.

### 3.1 Repository Structure

```
collab_nav_ground-jetson/     ← Jetson Orin Nano workspace
  src/
    collab_nav_uav/           ← UAV subsystem (git submodule → NoeRos22/collab_nav_uav)
      tello_driver/           ← DJI Tello ROS 2 driver
      tello_controller/       ← Manual keyboard control node
      tello_pos_control/      ← PID+feedforward position controller (OptiTrack-guided)
      tello_calibrator/       ← Camera intrinsic calibration pipeline
      tello_msgs/             ← Custom FlightStats / FlipControl messages
    arena_map_builder/        ← Aerial image stitching → occupancy grid
    arena_map_builder_msgs/   ← Service/message definitions for map builder
    arena_marker_localizer/   ← ArUco PnP pose estimation (service node)
    arena_marker_localizer_interfaces/
    aruco_localizer/          ← Lightweight ArUco detection node (AMR camera)
    amr_optitrack/            ← OptiTrack → /amr/pose bridge node
    amr_drone_nav/            ← World↔odom TF alignment; coordinate system glue
    map_fusion/               ← ICP-based fusion: aerial map + LiDAR SLAM map
    lidar_odometry/           ← Scan-matching LiDAR odometry / occupancy projection
    local_costmap/            ← Local costmap inflation around obstacles
    world_mapper/             ← Maintains running global occupancy map
    trajectory_planner/       ← A* path planner + cubic spline trajectory follower
    emergency_stop/           ← Automatic Emergency Braking (AEB) safety node
    frontier_explorer/        ← Fallback frontier-based exploration
    mission_orchestrator/     ← 20-stage mission state machine (overseer)
    optitrack_client/         ← VRPN→ROS OptiTrack client (git submodule)
    oradar_ros/               ← OraDAR MS200 LiDAR driver (git submodule)
    LightGlue-ONNX-Jetson/    ← SuperPoint+LightGlue ONNX inference (git submodule)
  network_bridge/             ← Security middleware: SecureEnvelope (git submodule)

collab_nav_ground-rasp/       ← Yahboom RDK X3 workspace
  src/
    ekf_amr/                  ← EKF sensor fusion (IMU + encoders + VSLAM VIO)
    amr_bringup/              ← Hardware bringup launch files for the RDK X3
```

### 3.2 Subsystem Decomposition

The system decomposes into five functional subsystems:

| Subsystem | Packages | Host |
|---|---|---|
| **UAV Flight** | `tello_driver`, `tello_pos_control`, `tello_controller`, `tello_calibrator` | Jetson |
| **Aerial Mapping** | `arena_map_builder`, `arena_marker_localizer`, `LightGlue-ONNX-Jetson` | Jetson |
| **AMR Localization** | `ekf_amr`, `aruco_localizer`, `amr_optitrack`, `lidar_odometry` | RDK X3 + Jetson |
| **Navigation & Planning** | `map_fusion`, `world_mapper`, `local_costmap`, `trajectory_planner`, `frontier_explorer` | Jetson |
| **Safety & Orchestration** | `emergency_stop`, `mission_orchestrator`, `amr_drone_nav` | Jetson |

---

## 4. End-to-End Pipeline

The full mission executes as a **20-stage sequential pipeline** managed by `mission_orchestrator`. Each stage must pass a quality gate before the next begins; failed stages trigger exponential-backoff retries.

```
Stage 1:   UAV takeoff and initialization
Stage 2:   OptiTrack lock confirmation
Stage 3:   Pre-scan arena position (hover)
Stage 4-N: Drone scan trajectory (lawnmower pattern)
Stage N+1: Land drone
Stage N+2: Stitch frames → aerial occupancy grid
Stage N+3: Localize ArUco markers (goal + AMR) in aerial image
Stage N+4: Compute world-frame goal pose
Stage N+5: Start AMR SLAM / lidar odometry
Stage N+6: Fuse aerial map with LiDAR SLAM map
Stage N+7: Plan A* path to goal
Stage N+8: Execute cubic-spline trajectory (with AEB active)
Stage N+9: Goal arrival confirmation
```

### 4.1 Step 1 — Drone Scan

The Tello executes a pre-planned **lawnmower trajectory** over the 4 × 4 m arena at a fixed altitude (~1.5 m). A **PID + feedforward** position controller (`tello_pos_control`) closes the loop using OptiTrack as the reference frame. Video frames are timestamped and streamed via the `tello_driver` ROS node.

### 4.2 Step 2 — Aerial Image Stitching (`arena_map_builder`)

Consecutive frames are sub-sampled and filtered, then aligned pairwise using either **SIFT** (CPU) or **SuperPoint + LightGlue** (GPU, via ONNX inference on the Jetson) descriptor matching followed by **RANSAC** homography estimation. A **pose graph** over all accepted pairwise transforms is optimized, and individual frames are composited using **Laplacian pyramid blending** to suppress seam artifacts. The resulting composite image is projected to a **ROS `OccupancyGrid`** with metric scale derived from the known arena dimensions and OptiTrack calibration.

### 4.3 Step 3 — ArUco Localization (`arena_marker_localizer`)

Both the drone's camera feed and the AMR's front camera feed are processed by a **PnP solver** (OpenCV `solvePnP`, IPPE or EPnP) applied to detected ArUco corners (OpenCV `aruco::detectMarkers`). This yields 6-DoF camera→marker transforms, which are composed with known camera extrinsics to obtain **world-frame poses** for the goal marker and the AMR fiducial.

### 4.4 Step 4 — Map Fusion (`map_fusion`)

The drone-derived aerial occupancy grid and the AMR's LiDAR-built occupancy grid (generated by `lidar_odometry` scan-matching) are registered to a common frame using a **coarse search** (correlation or grid-search) followed by **ICP (Iterative Closest Point)** refinement on the extracted obstacle point clouds. The fused grid is an **augmented probabilistic occupancy map** that compensates for individual sensor drift.

### 4.5 Step 5 — Path Planning & Trajectory Execution (`trajectory_planner`)

An **A\* search** with **octile heuristic** (for diagonal movement on the grid) plans a collision-free path on the fused occupancy grid. The discrete waypoints are smoothed into a **parametric cubic spline** trajectory. A spline follower node converts the spline to velocity commands for the AMR drive. At all times, the **AEB node** (`emergency_stop`) monitors LiDAR returns and overrides the velocity command if an obstacle is closer than a configurable safety threshold.

### 4.6 AMR Odometry & Localization (`ekf_amr`)

The AMR maintains a continuous pose estimate via an **Extended Kalman Filter** fusing three sources:
- **Wheel encoder odometry** (dead-reckoning, high-rate, accumulates drift)
- **IMU** (accelerometer + gyroscope, corrects short-term angular drift)
- **Isaac VSLAM Visual-Inertial Odometry** (camera + IMU, long-term drift correction)

The EKF state is `[x, y, θ, vx, vy, ω]` in the odometry frame. World-frame anchoring is provided by ArUco PnP fixes when markers are visible, or by the OptiTrack-derived pose when available.

---

## 5. Security Architecture (`network_bridge`)

Every ROS 2 topic that carries mission-critical data (pose estimates, velocity commands, map data) is optionally wrapped in a **SecureEnvelope** at the application layer. This provides security guarantees orthogonal to DDS SROS2, and operates as a transparent middleware:

```
Standard:  Node A  ──[ROS 2 topic]──►  Node B

Secured:   Node A  ──► SecureEnvelope ──► Node B
                        ├─ RSA-PSS signature (sender identity + integrity)
                        ├─ AES-256-GCM encryption (optional, per-topic policy)
                        └─ Timestamp window (replay attack prevention)
```

Per-node security policy is configured in YAML. A global kill-switch allows bypassing the envelope and reverting to native ROS 2 for debugging.

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| Jetson as compute hub, RDK X3 as drive platform | Separates high-power compute (stitching, planning) from low-latency motor control |
| OptiTrack as GPS surrogate | Ground-level GPS is unreliable indoors; OptiTrack gives sub-mm accuracy within the test volume |
| SIFT + SuperPoint dual-mode stitching | SIFT as a reliable CPU fallback; SuperPoint+LightGlue for higher accuracy when Jetson GPU is available |
| Laplacian pyramid blending | Preserves high-frequency detail (obstacle edges, ArUco corners) better than simple alpha blending |
| ICP map fusion (not just overlay) | Compensates for accumulated drift between the drone reference frame and AMR odometry frame |
| 20-stage orchestrator with health checks | Enables autonomous unsupervised execution; per-stage quality gates prevent silent propagation of bad data |
| AEB as independent safety layer | Runs at higher frequency than the planner; decoupled from planning failures |
| Frontier explorer as fallback | If stitching or localization fails, the AMR can still explore to find the goal marker ground-truth ArUco |

---

## 7. Report Outline (Proposed)

Per the professor's proposed structure, the final written report will cover:

1. **Theoretical Framework** — per-subsystem theory sections
   - Kinematics of differential-drive AMR
   - Extended Kalman Filter for sensor fusion
   - Feature detection and matching (SIFT, SuperPoint, LightGlue)
   - Image stitching: homography estimation, RANSAC, pose graphs, Laplacian blending
   - ArUco marker detection and PnP pose estimation
   - Occupancy grids and probabilistic mapping
   - A* path planning and spline trajectory generation
   - ICP point-cloud registration

2. **Requirements** — mechanical, electronic, functional

3. **Development Methodology** — incremental stages 1→4, CI/test strategy

4. **System Architecture & Subsystems** (this document + per-package `DOCS.md` files)

5. **Detailed Subsystem Documentation** — requirements, traceability matrix, implementation, tests

6. **Experimental Results** — demo video, metric tables, stitching quality, localization accuracy, navigation success rate

7. **Code & Evidence References** — links to per-package DOCS.md, test runs

---

## 8. Package Documentation Index

Each package has (or will have) its own `DOCS.md` at `src/<package>/DOCS.md`:

| Package | Domain | DOCS.md |
|---|---|---|
| `tello_pos_control` | UAV flight control | [link](src/collab_nav_uav/src/tello_pos_control/DOCS.md) |
| `tello_driver` | UAV ROS bridge | [link](src/collab_nav_uav/src/tello_driver/DOCS.md) |
| `arena_map_builder` | Aerial stitching | [link](src/arena_map_builder/DOCS.md) |
| `arena_marker_localizer` | ArUco PnP localization | [link](src/arena_marker_localizer/DOCS.md) |
| `map_fusion` | Map registration & fusion | [link](src/map_fusion/DOCS.md) |
| `trajectory_planner` | A* + spline planning | [link](src/trajectory_planner/DOCS.md) |
| `emergency_stop` | AEB safety | [link](src/emergency_stop/DOCS.md) |
| `mission_orchestrator` | Mission state machine | [link](src/mission_orchestrator/DOCS.md) |
| `ekf_amr` | EKF odometry (RDK X3) | [link](../../collab_nav_ground-rasp/src/ekf_amr/DOCS.md) |
| `lidar_odometry` | LiDAR scan-matching | [link](src/lidar_odometry/DOCS.md) |
| `frontier_explorer` | Fallback exploration | [link](src/frontier_explorer/DOCS.md) |
| `network_bridge` | Security middleware | [link](network_bridge/DOCS.md) |
