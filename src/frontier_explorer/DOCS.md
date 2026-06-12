# Frontier Explorer
## Technical Documentation

---

## 1. Overview

`frontier_explorer` is a ROS 2 package that implements the **fallback navigation strategy** for the arena challenge mission. It activates when the drone-based map pipeline (`arena_map_builder`) fails to produce a usable occupancy grid — leaving the AMR with no knowledge of the environment and no world-frame position for the goal ArUco marker.

In this scenario the package drives the AMR through the arena autonomously using **frontier-based exploration** on its onboard SLAM map. It continuously scans for the target ArUco marker with the RGB camera. The moment the marker is detected, the exploration phase ends and the robot homes directly to it using either A\*-planned trajectories or a pure visual servo controller.

**Pipeline summary** (four cooperating nodes):

```
(1) ArucoGoalDetector  →  raw marker pose in camera frame
(2) ArucoWorldBridge   →  transforms pose into world frame
(3) FrontierExplorer   →  selects best unexplored frontier as navigation goal
(4) ExplorerController →  state machine that arbitrates between (3) and (2)
```

An optional fifth node, `CameraFovTracker`, builds a persistent map of which cells the RGB camera has ever swept over. When enabled, `FrontierExplorer` filters out frontier clusters whose adjacent unknown space has already been seen by the camera, preventing the robot from re-exploring areas the camera has fully covered.

**Trigger condition:** the `ExplorerController` sits in `IDLE` until the operator (or the broader mission orchestrator) publishes a single boolean to `/map_fail_fallback/start`. Everything else is autonomous.

---

## 2. System Context and Prerequisites

The package is a **fallback stack** — it does not run in isolation. The following nodes must already be running and providing a complete TF tree before `frontier_explorer_launch.py` is started:

| Dependency | What it provides |
|---|---|
| `occupancy_mapper` | `/amr/world_map` (nav_msgs/OccupancyGrid) — the SLAM map |
| EKF node | `/amr/ekf/odom` (nav_msgs/Odometry) + TF `odom→base_footprint` |
| Localisation node | TF `world→odom` |
| D435i RealSense driver | `/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`, camera TF subtree |
| `trajectory_planner` | `astar_planner2` + `spline_follower`, subscribes `/aruco/goal/pose` |
| AMR base controller | Subscribes `/amr/reference` (nav_msgs/Odometry) for velocity commands |

**Required TF chain:**
```
world → odom → base_footprint → camera_link → camera_color_optical_frame
                              → lidar
```

Verify the tree is complete before starting the mission:
```bash
ros2 run tf2_ros tf2_echo world base_footprint
```

---

## 3. Node Architecture

### 3.1 ExplorerController (`explorer_controller`)

The central state machine. All other nodes in the package are subordinate to it. It owns the lifecycle of the exploration mission and arbitrates which goal is forwarded to the path planner.

**States:**

```
IDLE  ──(operator start)──►  EXPLORING  ──(ArUco detected)──►  HOMING  ──(reached)──►  DONE
                                 ▲                                  │
                                 └──────────(detection lost)────────┘
```

| State | Behaviour |
|---|---|
| `IDLE` | All nodes silent. Waits for `/map_fail_fallback/start = true`. |
| `EXPLORING` | Activates `FrontierExplorer`. Forwards `/frontier/goal` to `/aruco/goal/pose` (consumed by A\*). Progress watchdog re-sends the goal if the robot stalls for more than 4 s. |
| `HOMING` | Silences `FrontierExplorer`. Last known world-frame ArUco pose forwarded to A\* every tick. Falls back to `EXPLORING` if the detection goes stale (`detection_timeout_sec`). |
| `DONE` | Terminal. Everything stops. Node must be restarted to run again. |

The controller also handles A\* failures: after two consecutive planning failures on the same frontier goal, it publishes a blacklist signal back to `FrontierExplorer` so that cluster is excluded from future scoring.

A progress watchdog inside `EXPLORING` detects when the robot has not moved `0.05 m` in `4.0 s` and re-sends the current goal. After two such retries with no movement, it treats the goal as unreachable and forces `FrontierExplorer` to select a new one.

**Subscriptions:**

| Topic | Type | Purpose |
|---|---|---|
| `/map_fail_fallback/start` | `std_msgs/Bool` | Operator trigger |
| `/frontier/goal` | `geometry_msgs/PoseWithCovarianceStamped` | Goal from FrontierExplorer |
| `/aruco/detection` | `geometry_msgs/PoseWithCovarianceStamped` | World-frame marker pose from ArucoWorldBridge |
| `/follower/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Robot pose |
| `/astar/goal_failed` | `geometry_msgs/PoseWithCovarianceStamped` | A\* planning failure signal |
| `/aruco_servo/active` | `std_msgs/Bool` | Visual servo completion signal |

**Publications:**

| Topic | Type | Purpose |
|---|---|---|
| `/aruco/goal/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Goal forwarded to A\* planner |
| `/frontier_explorer/active` | `std_msgs/Bool` | Activates or silences FrontierExplorer |
| `/aruco_servo/enable` | `std_msgs/Bool` | Enables or disables ArucoVisualServo |
| `/trajectory_planner2/path` | `nav_msgs/Path` | Empty path sent to cancel spline follower on state transitions |
| `/mission/state` | `std_msgs/String` | Current state name for monitoring |
| `/mission/status_marker` | `visualization_msgs/Marker` | RViz text overlay |

---

### 3.2 FrontierExplorer (`frontier_explorer`)

Detects frontiers on the occupancy map, clusters them, scores each cluster, and publishes the best centroid as a navigation goal. A **frontier cell** is any `FREE` cell (`0 ≤ value < 90`) that has at least one `UNKNOWN` neighbour (`value == -1`). Clusters are built with a union-find over 8-connected frontier cells.

The node starts silent (`active = False`) and is activated by `ExplorerController` via `/frontier_explorer/active`. It publishes nothing while inactive.

**Scoring (higher is better):**

$$\text{score} = w_{\text{dist}} \cdot \text{norm\_prox} + w_{\text{size}} \cdot \text{norm\_area}$$

Both terms are normalised to $[0, 1]$ across the candidate set before combining. Proximity is derived from a *heading-inflated perceived distance* rather than raw Euclidean distance: the bearing from the robot to the cluster centroid is compared against the robot's current heading, and off-axis clusters are penalised exponentially:

$$\text{heading\_score} = \left(\frac{1 + \cos(\Delta\theta)}{2}\right)^3$$

Clusters more than 60° off the robot's heading have their heading score multiplied by 0.01, making them effectively invisible to the scorer unless no forward-facing candidates exist. This prevents thrashing U-turns between frontiers on opposite sides of the robot.

**Centroid safety walk:** after the best centroid is selected, the node walks it stepwise toward the robot along the straight-line vector until it finds a cell that is free in the SLAM map and clear of obstacles within `safe_goal_radius`. This prevents goals from being placed inside or adjacent to lethal inflation zones that A\* cannot plan through.

**Goal blacklisting:** goals for which A\* reports a planning failure (`/astar/goal_failed`) are added to a failure blacklist with radius `0.60 m`. Goals that have been successfully reached are added to a temporal reached-blacklist (`0.20 m` radius, 30 s TTL) to prevent immediately revisiting the same frontier.

**Subscriptions:**

| Topic | Type | Purpose |
|---|---|---|
| `/drone/map` (configurable) | `nav_msgs/OccupancyGrid` | Occupancy map from SLAM |
| `/follower/pose` (sim) or `odom_topic` (real) | `PoseWithCovarianceStamped` / `Odometry` | Robot pose |
| `/frontier_explorer/active` | `std_msgs/Bool` | Activation signal from ExplorerController |
| `/astar/goal_failed` | `geometry_msgs/PoseWithCovarianceStamped` | Blacklist trigger |
| `/camera/fov_map` (optional) | `nav_msgs/OccupancyGrid` | Camera coverage mask from CameraFovTracker |

**Publications:**

| Topic | Type | Purpose |
|---|---|---|
| `/frontier/goal` | `geometry_msgs/PoseWithCovarianceStamped` | Best frontier centroid |
| `/frontier/markers` | `visualization_msgs/MarkerArray` | Frontier cells and cluster centroids for RViz |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `map_topic` | `/drone/map` | Occupancy map topic |
| `pose_topic` | `/follower/pose` | Pose topic (simulation mode) |
| `odom_topic` | `''` | EKF odometry topic (real robot mode; set to activate TF transform path) |
| `world_frame` | `'map'` | Fixed frame |
| `min_cluster_size` | `5` cells | Discard smaller frontier clusters |
| `max_frontier_dist` | `10.0` m | Ignore frontiers beyond this distance |
| `update_rate` | `1.0` Hz | Frontier recomputation frequency |
| `goal_reached_dist` | `0.12` m | Distance at which the robot is considered to have reached a frontier goal |
| `min_goal_dist` | `0.20` m | Ignore frontiers closer than this |
| `safe_goal_radius` | `0.30` m | Obstacle clearance radius for the safety walk |
| `w_dist` | `0.85` | Scoring weight for heading-inflated proximity |
| `w_size` | `0.15` | Scoring weight for cluster area (m²) |
| `require_camera_coverage` | `False` | If `True`, skip frontier clusters whose unknown neighbours have already been seen by the camera |
| `fov_map_topic` | `/camera/fov_map` | Camera coverage mask topic (only used when `require_camera_coverage = True`) |

---

### 3.3 ArucoGoalDetector (`aruco_goal_detector`)

Detects ArUco markers from the RGB camera stream using OpenCV's `aruco.detectMarkers` + `solvePnP`. Publishes each detected marker's 6-DoF pose relative to the camera frame.

If the same marker ID appears multiple times in one frame (e.g. two faces of a cube both visible), the translation and rotation vectors are averaged across all detections before publishing. Tiny detections below `min_detection_area` pixels² are discarded as noise.

**Subscriptions:**

| Topic | Type |
|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |

**Publications:**

| Topic | Type | Description |
|---|---|---|
| `/aruco/detections` | `geometry_msgs/PoseArray` | All markers detected this frame (camera frame) |
| `/aruco/id_{id}/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Per-marker-ID pose (camera frame) |
| `/aruco/markers` | `visualization_msgs/MarkerArray` | RViz cube overlays |
| `/aruco/debug_image` | `sensor_msgs/Image` | Annotated image with axes drawn (when `publish_debug_image = True`) |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `marker_size_m` | `0.13` m | Physical side length of the ArUco marker |
| `camera_frame` | `'camera_color_optical_frame'` | Frame for published poses |
| `image_topic` | `/camera/camera/color/image_raw` | |
| `camera_info_topic` | `/camera/camera/color/camera_info` | |
| `aruco_dict` | `'DICT_4X4_50'` | ArUco dictionary; valid options: `DICT_4X4_{50,100,250}`, `DICT_5X5_{50,100}`, `DICT_6X6_50`, `DICT_ARUCO_ORIGINAL` |
| `min_detection_area` | `100` px² | Minimum marker corner area; increase to suppress false positives at range |
| `publish_debug_image` | `True` | Publish annotated image to `/aruco/debug_image` |

---

### 3.4 ArucoWorldBridge (`aruco_world_bridge`)

A thin TF bridge that converts the per-ID camera-frame pose published by `ArucoGoalDetector` into the world frame expected by `ExplorerController`. It subscribes to `/aruco/id_{target_marker_id}/pose` and republishes as `/aruco/detection` with the pose in world frame and the marker ID encoded in `covariance[0]` (a convention shared with the simulation's `fake_aruco_detector`).

**Subscriptions:**

| Topic | Type |
|---|---|
| `/aruco/id_{target_marker_id}/pose` | `geometry_msgs/PoseWithCovarianceStamped` |

**Publications:**

| Topic | Type |
|---|---|
| `/aruco/detection` | `geometry_msgs/PoseWithCovarianceStamped` |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `target_marker_id` | `0` | ArUco ID to bridge; must match `target_marker_id` in ExplorerController |
| `world_frame` | `'map'` | Target TF frame |
| `tf_timeout_sec` | `0.1` s | TF lookup timeout |

---

### 3.5 ArucoVisualServo (`aruco_visual_servo`)

A direct image-space visual servo controller. It activates when `ExplorerController` publishes `/aruco_servo/enable = True` (i.e. when the state transitions to `HOMING`) and takes exclusive control of `/amr/reference` — the same velocity topic used by the spline follower. It requires no map, no EKF, and no TF beyond what is already available; it drives purely from the camera image.

**Control law** (runs at `update_rate` Hz):

The normalised horizontal pixel error is:

$$e_x = \frac{t_x / t_z}{\tan(\text{hfov}/2)} \in [-1, +1]$$

where $t_x$ is the marker's horizontal offset in the camera frame and $t_z$ is depth. Positive $e_x$ means the marker is to the right of the image centre.

The velocity commands are:

$$\omega = -K_w \cdot e_x \quad \text{(angular)}$$

$$v = K_v \cdot (t_z - d_{\text{stop}}) \cdot f_{\text{centre}} \quad \text{(linear)}$$

where $f_{\text{centre}}$ is a binary centering gate: forward motion is enabled only when $|e_x| \leq \texttt{centering\_threshold}$. This ensures the robot centres on the marker horizontally before advancing, avoiding overshoot past it.

The node signals completion (`/aruco_servo/active = False`) when the marker depth $t_z \leq d_{\text{stop}}$ (goal reached) or when no detection arrives for `timeout_sec` seconds (detection lost). `ExplorerController` handles both outcomes.

**Subscriptions:**

| Topic | Type |
|---|---|
| `/aruco/id_{target_marker_id}/pose` | `geometry_msgs/PoseWithCovarianceStamped` |
| `/aruco_servo/enable` | `std_msgs/Bool` |

**Publications:**

| Topic | Type |
|---|---|
| `/amr/reference` | `nav_msgs/Odometry` |
| `/aruco_servo/active` | `std_msgs/Bool` |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `target_marker_id` | `5` | ArUco ID to servo toward |
| `stop_dist_m` | `0.50` m | Stop this far from the marker face |
| `Kw` | `0.60` | Angular gain (rad/s per normalised pixel) |
| `Kv` | `0.40` | Linear gain (m/s per metre) |
| `max_linear` | `0.25` m/s | Forward speed cap |
| `max_angular` | `0.60` rad/s | Rotation speed cap |
| `centering_threshold` | `0.10` | $|e_x|$ below which forward motion is enabled |
| `timeout_sec` | `2.0` s | Detection-lost timeout |
| `tan_hfov_half` | `0.693` | $\tan(\text{hfov}/2)$ for the D435i at 69.4°; override for a different camera |
| `update_rate` | `20.0` Hz | Servo loop frequency |

---

### 3.6 CameraFovTracker (`camera_fov_tracker`) — Optional

Tracks which cells of the occupancy map have ever been seen by the RGB camera. Publishes the result as a persistent boolean `OccupancyGrid` (`0 = unseen`, `100 = seen`) consumed by `FrontierExplorer` when `require_camera_coverage = True`.

**How it works:** every tick, the node looks up the camera pose in world frame via TF, computes the four ground-plane rays that correspond to the image corners using the camera intrinsic matrix, finds where each ray intersects $z = 0$ (the floor plane), and rasterises the resulting frustum polygon into the map grid with OpenCV `fillPoly`. The result is OR'd into a persistent boolean mask that monotonically grows over the mission.

> **Important:** this node requires the camera to be **pitched downward** so that its FOV rays intersect the floor plane. With a horizontally-mounted camera (the default in `frontier_explorer_launch.py`), the rays never hit $z = 0$ and the node produces only warnings. It is conditionally launched via `IfCondition(require_camera_coverage)` — leave that argument `false` unless the camera is pitched.

**Subscriptions:**

| Topic | Type |
|---|---|
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| `/slam/map` (configurable) | `nav_msgs/OccupancyGrid` |
| `/amr/ekf/odom` (configurable) | `nav_msgs/Odometry` |

**Publications:**

| Topic | Type |
|---|---|
| `/camera/fov_map` | `nav_msgs/OccupancyGrid` |
| `/camera/fov_marker` | `visualization_msgs/Marker` |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `camera_frame` | `'camera_color_optical_frame'` | |
| `world_frame` | `'world'` | |
| `camera_info_topic` | `/camera/camera/color/camera_info` | |
| `map_topic` | `/slam/map` | |
| `odom_topic` | `/amr/ekf/odom` | Triggers the tick on each odom message |
| `fov_map_topic` | `/camera/fov_map` | Output topic |
| `fov_marker_topic` | `/camera/fov_marker` | RViz polygon topic |
| `max_ray_length_m` | `8.0` m | Clip floor-intersection rays to this length |
| `update_rate_hz` | `5.0` Hz | FOV reprojection frequency |
| `tf_timeout_sec` | `0.1` s | TF lookup timeout |

---

## 4. Topic Graph

```
                  ┌─────────────────────────────────────────┐
                  │          ExplorerController              │
                  │           (state machine)                │
                  └────┬──────────┬──────────────┬──────────┘
                       │          │              │
          /frontier_   │    /aruco_servo/   /aruco/
          explorer/    │    enable          goal/pose
          active       │          │              │
                       ▼          ▼              ▼
              ┌─────────────┐  ┌────────────┐  ┌──────────────┐
              │  Frontier   │  │   Aruco    │  │ trajectory_  │
              │  Explorer   │  │ VisualServo│  │ planner /    │
              └──────┬──────┘  └─────┬──────┘  │ astar + spline│
                     │               │          └──────────────┘
             /frontier/goal  /amr/reference
                     │
                     ▼
              ExplorerController ──► /aruco/goal/pose ──► trajectory_planner

  ┌─────────────────┐    /aruco/id_{id}/pose    ┌──────────────────┐
  │  ArucoGoal      │ ──────────────────────────►│  ArucoWorld      │
  │  Detector       │  (camera frame)            │  Bridge          │
  │                 │ ──────────────────────────►│                  │
  └─────────────────┘                            └────────┬─────────┘
    camera/image_raw                                      │ /aruco/detection
    camera/camera_info                                    │ (world frame)
                                                          ▼
                                                 ExplorerController

  ┌─────────────────┐    /camera/fov_map
  │  CameraFov      │ ──────────────────► FrontierExplorer
  │  Tracker        │  (optional)
  └─────────────────┘
```

---

## 5. Launch File

### 5.1 Real-Robot Launch

```bash
ros2 launch frontier_explorer frontier_explorer_launch.py
```

Start the mission after confirming localisation is running:

```bash
ros2 topic pub /map_fail_fallback/start std_msgs/msg/Bool "{data: true}" --once
```

Reset back to `IDLE` (aborts exploration without restarting the node):

```bash
ros2 topic pub /map_fail_fallback/start std_msgs/msg/Bool "{data: false}" --once
```

### 5.2 Launch Arguments

**Frames**

| Argument | Default | Description |
|---|---|---|
| `world_frame` | `'world'` | Fixed world frame — must match `occupancy_mapper world_frame` |
| `camera_frame` | `'camera_color_optical_frame'` | Camera optical frame |

**Topics**

| Argument | Default | Description |
|---|---|---|
| `odom_topic` | `/amr/ekf/odom` | EKF odometry — transformed to world frame via TF by FrontierExplorer |
| `map_topic` | `/amr/world_map` | Occupancy map from `occupancy_mapper` |

**Camera mount** (static TF `base_footprint → camera_link`)

| Argument | Default | Notes |
|---|---|---|
| `camera_x` | `0.1` m | |
| `camera_y` | `0.0` m | |
| `camera_z` | `0.2` m | |
| `camera_roll` | `-1.5708` rad | Compensates RealSense optical frame rotation |
| `camera_pitch` | `0.0` rad | |
| `camera_yaw` | `-1.5708` rad | Compensates RealSense optical frame rotation |

**ArUco**

| Argument | Default | Description |
|---|---|---|
| `target_marker_id` | `4` | ArUco ID of the mission goal marker |
| `marker_size_m` | `0.13` m | Physical side length |
| `aruco_dict` | `'DICT_4X4_50'` | |
| `min_detection_area` | `200` px² | |

**Frontier explorer**

| Argument | Default | Description |
|---|---|---|
| `w_dist` | `0.85` | Proximity weight |
| `w_size` | `0.15` | Cluster area weight |
| `min_cluster_size` | `5` cells | |
| `max_frontier_dist` | `8.0` m | |
| `frontier_update_rate` | `1.0` Hz | |
| `min_goal_dist` | `0.40` m | |
| `frontier_goal_reached_dist` | `0.20` m | |
| `safe_goal_radius` | `0.25` m | Obstacle clearance radius for goal safety walk |
| `require_camera_coverage` | `false` | Set `true` only if camera is pitched downward (also launches `camera_fov_tracker`) |

**ExplorerController**

| Argument | Default | Description |
|---|---|---|
| `goal_reached_dist` | `0.35` m | Distance to ArUco goal that triggers `DONE` |
| `detection_timeout_sec` | `2.0` s | Detection gap before falling back from `HOMING` to `EXPLORING` |

**Misc**

| Argument | Default | Description |
|---|---|---|
| `fov_update_rate` | `5.0` Hz | `CameraFovTracker` update rate (only used when `require_camera_coverage = true`) |
| `rviz` | `true` | Launch RViz with the bundled config |

### 5.3 Nodes Launched

| Node name | Executable | Always launched |
|---|---|---|
| `camera_link_tf` | `static_transform_publisher` | Yes |
| `aruco_goal_detector` | `aruco_goal_detector` | Yes |
| `aruco_world_bridge` | `aruco_world_bridge` | Yes |
| `camera_fov_tracker` | `camera_fov_tracker` | Only if `require_camera_coverage = true` |
| `frontier_explorer` | `frontier_explorer` | Yes (starts inactive) |
| `explorer_controller` | `explorer_controller` | Yes |
| `aruco_visual_servo` | `aruco_visual_servo` | Yes |
| `rviz2` | `rviz2` | Only if `rviz = true` |

> **Note on `world_odom_tf`:** the launch file contains a commented-out static `world→odom` identity transform. Uncomment it only if no external localisation node is publishing that transform — otherwise it will conflict with the real transform.

---

## 6. Frontier Detection and Scoring Algorithm

### 6.1 Step 1 — Detect Frontier Cells

The SLAM map is analysed as a 2D NumPy array. A cell is classified as a frontier cell iff it satisfies both conditions simultaneously:

- **Free:** `0 ≤ value < 90` (covers both the canonical `0` and the low positive values `1–49` that occupancy mappers emit for ray-traced free space)
- **Borders unknown:** at least one 4-connected neighbour has `value == -1`

The detection is implemented as a pair of boolean masks with `np.roll`-based neighbour shifts — no Python loops over the grid, making it efficient on the 80×80 cells typical of this arena.

### 6.2 Step 2 — Cluster with Union-Find

Frontier cells are grouped into connected components using a **union-find** (disjoint-set) structure over 8-connected adjacency. Path compression is applied. Clusters with fewer than `min_cluster_size` cells are discarded.

### 6.3 Step 3 — Score and Select

For each surviving cluster:

1. Compute the **safe centroid** — the arithmetic mean of all cell indices, snapped to the nearest real frontier cell if the mean falls on a non-free cell.
2. Run the **safety walk** to find the nearest obstacle-clear point within `safe_goal_radius` of the centroid.
3. Check against the failure blacklist and the temporal reached-blacklist.
4. Compute the **heading-inflated perceived distance** and score with the weighted linear combination.

The cluster with the highest score is selected as the current goal. The goal is published once on change; it is re-published on subsequent ticks only if `ExplorerController` resets it (after an A\* failure) or if the robot has not yet arrived.

---

## 7. Design Notes

**Why the heading-inflation approach rather than a separate heading term?**
Normalising a separate heading term against the full candidate set collapses its contribution whenever all frontiers face similar directions (range shrinks to near zero). Baking heading into the effective distance keeps the penalty meaningful regardless of the candidate distribution — a behind-facing frontier always perceives itself as far, even when it is the closest one geometrically.

**Why the 60° hard cutoff with a ×0.01 multiplier?**
The cubic cosine term is smooth but decays slowly through 60–90°. Without the cutoff, the robot frequently considered sideways frontiers competitive enough to trigger heading changes when a forward cluster of similar quality existed. The ×0.01 floor (rather than zero) ensures the robot can still escape if every remaining frontier is behind it.

**Why is ArucoVisualServo separate from A\*-based homing?**
During `HOMING`, A\* plans a path to the last *known* world-frame position of the marker, which may have been observed from several metres away. Localisation drift and the discrete resolution of the occupancy grid mean the planned endpoint may miss the marker by 10–20 cm. The visual servo closes that last gap purely from image-space feedback, independent of any world-model error.

**Why does FrontierExplorer start inactive?**
The node is always launched but starts with `active = False`. `ExplorerController` manages its lifecycle via `/frontier_explorer/active`. This avoids a race condition where `FrontierExplorer` publishes a goal before the state machine is ready to handle it, and allows `ExplorerController` to cleanly silence it during `HOMING` and `DONE` without node teardown.

---

## 8. Package Structure

```
frontier_explorer/
├── frontier_explorer/
│   ├── frontier_explorer.py        # FrontierExplorer node
│   ├── explorer_controller.py      # ExplorerController (state machine)
│   ├── aruco_goal_detector.py      # ArUco detector (camera frame)
│   ├── aruco_world_bridge.py       # Camera → world frame TF bridge
│   ├── aruco_visual_servo.py       # Image-space visual servo
│   ├── camera_fov_tracker.py       # Camera FOV coverage mask (optional)
│   └── simulation_helpers/         # Simulation-only helper nodes
├── launch/
│   ├── frontier_explorer_launch.py # Real-robot launch (primary)
│   └── explore_sim_launch.py       # Simulation launch
├── rviz/
│   └── frontier_exploration.rviz   # RViz configuration
├── package.xml
└── setup.py
```

---

## 9. ROS 2 Package Dependencies

Declared in `package.xml`:

| Package | Use |
|---|---|
| `rclpy` | Node framework |
| `nav_msgs` | `OccupancyGrid`, `Odometry`, `Path` |
| `geometry_msgs` | `PoseWithCovarianceStamped`, `PoseStamped`, `Point` |
| `sensor_msgs` | `Image`, `CameraInfo` |
| `std_msgs` | `Bool`, `String` |
| `visualization_msgs` | `Marker`, `MarkerArray` |

Additional runtime Python dependencies (not in `package.xml`, must be available in the environment):

| Library | Use |
|---|---|
| `opencv-python` | ArUco detection, solvePnP, FOV polygon rasterisation |
| `cv_bridge` | ROS image ↔ OpenCV conversion |
| `numpy` | Map array operations |
| `tf2_ros`, `tf2_geometry_msgs` | TF lookups and pose transforms |