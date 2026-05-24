# mission_orchestrator

ROS 2 (Humble) package that bootstraps and sequences the full
**collab\_nav** ground-robot + drone pipeline from a single launch command.

---

## Where it fits

The collab\_nav system has two physical platforms:

| Platform | Compute | Packages |
|---|---|---|
| Ground robot (AMR) | Raspberry Pi | `amr_bringup`, `amr_ekf`, `oradar_ros`, `map_fusion`, `trajectory_planner` |
| Drone (Tello) | Jetson (this machine) | `tello_driver`, `tello_pos_control`, `arena_marker_localizer`, `arena_map_builder` |

`mission_orchestrator` runs **on the Jetson** and is the single entry
point that starts all other subsystems in the right order, checks their
health, and hands off control once the arena map is live.

```
                        ┌──────────────────────────────────────┐
                        │         mission_orchestrator          │
                        │                                       │
  Raspberry Pi ←─ SSH ──┤  stages 01-04: AMR bring-up + EKF   │
  OptiTrack   ──────────┤  stage  05:    pose stream check     │
  Tello drone ──────────┤  stages 06-11: flight + video ingest │
  Jetson (local) ───────┤  stages 12-20: map + planning launch │
                        └──────────────────────────────────────┘
```

After stage 20 the orchestrator goes idle; `map_fusion` and
`trajectory_planner` operate autonomously.

---

## Pipeline — 20 stages

| # | What happens | Abort condition |
|---|---|---|
| 01 | Ping Raspberry Pi | Host unreachable |
| 02 | SSH connect to Raspberry Pi | Auth / timeout failure |
| 03 | Launch AMR bringup via SSH (`nohup`); capture remote PID | — |
| 04 | Wait for `/amr/ekf/odom` to stabilise (sliding-window mean \|v\| < threshold) | EKF timeout |
| 05 | Verify `/optitrack/rigid_body`: frame\_id + stamp freshness; auto-launch client if silent | No message / bad header / stale stamp |
| 06 | Launch `tello_driver` | — |
| 07 | Drone preflight: `/camera/image_raw` live, `/battery_state` ≥ minimum % | Camera timeout / low battery |
| 08 | Launch `tello_map` (drone takes off and executes scanning routine) | — |
| 09 | Monitor drone state transitions **1 → 2 → 3 → 4** with per-state timeouts | Any state timeout |
| 10 | Wait for `/drone/video_filename` and `/drone/telemetry_filename` topics | Topic timeout |
| 11 | Verify `scan.mp4` integrity with ffmpeg | Corrupt video |
| 12 | Launch `trajectory_planner` | — |
| 13 | Launch `map_fusion` | — |
| 14 | Launch `oradar` lidar | — |
| 15 | Launch `arena_marker_localizer` service node; wait for `/localize_markers` | Service never ready |
| 16 | Call `/localize_markers` (video + telemetry CSV → marker poses) | Service failure |
| 17 | Publish `/aruco/amr/pose` and `/aruco/goal/pose` as `PoseWithCovarianceStamped` (latched) | Missing marker IDs |
| 18 | Launch `arena_map_builder` server; set `background_path` parameter | Server never ready |
| 19 | Send `BuildArenaMap` action goal; wait for result | Action failure / timeout |
| 20 | Publish `OccupancyGrid` result to `/drone/map` (latched) | — |

### Drone state machine

`tello_pos_control` publishes the drone's internal state on `/drone/state`
(`std_msgs/Int32`). The orchestrator monitors these values:

| Value | Meaning |
|---|---|
| −1 | Before takeoff (initial) |
| 0 | Lifting off |
| 1 | Stabilize |
| 2 | Executing trajectory |
| 3 | Returning to home |
| 4 | Landing |

Stage 09 waits for states 1 → 2 → 3 → 4 in order.
Stage 07 asserts the state is still −1 (or not yet received) before arming.

---

## ROS 2 interfaces

### Subscribed topics

| Topic | Type | Used by |
|---|---|---|
| `/amr/ekf/odom` | `nav_msgs/Odometry` | Stage 04 — EKF stability check |
| `/optitrack/rigid_body` | `geometry_msgs/PoseStamped` | Stage 05 — OptiTrack health |
| `/drone/state` | `std_msgs/Int32` | Stages 07, 09 |
| `/camera/image_raw` | `sensor_msgs/Image` | Stage 07 — camera preflight |
| `/battery_state` | `sensor_msgs/BatteryState` | Stage 07 — battery preflight |
| `/drone/video_filename` | `std_msgs/String` | Stage 10 — path to scan.mp4 |
| `/drone/telemetry_filename` | `std_msgs/String` | Stage 10 — path to telemetry.csv |
| `/drone/map` | `nav_msgs/OccupancyGrid` | Stage 20 — echo-back confirmation |

### Published topics

| Topic | Type | QoS | Purpose |
|---|---|---|---|
| `/aruco/amr/pose` | `geometry_msgs/PoseWithCovarianceStamped` | latched | AMR initial pose for `map_fusion` / `slam_toolbox` |
| `/aruco/goal/pose` | `geometry_msgs/PoseWithCovarianceStamped` | latched | Goal pose for `trajectory_planner` |
| `/drone/map` | `nav_msgs/OccupancyGrid` | latched | Arena occupancy grid for all navigation stack consumers |
| `/cmd_vel` | `geometry_msgs/Twist` | volatile | Zero velocity command during drone abort |

### Service client

| Service | Type | Used by |
|---|---|---|
| `/localize_markers` | `arena_marker_localizer_interfaces/srv/LocalizeMarkers` | Stage 16 |

### Action client

| Action | Type | Used by |
|---|---|---|
| `build_arena_map` | `arena_map_builder_msgs/action/BuildArenaMap` | Stages 18–19 |

---

## Configuration

All parameters live in `config/orchestrator_params.yaml`.
The file is loaded at runtime via the `config_file` ROS parameter —
**no recompile needed** to change values.

### Key sections

```yaml
orchestrator:

  rasp:
    ip: "10.42.0.184"          # Raspberry Pi address
    user: "root"
    password: "root"
    amr_launch_cmd: >-         # Command executed over SSH
      source ~/ros2_ws/install/setup.bash &&
      nohup ros2 launch amr_bringup launch_rasp.py
      > /tmp/amr_launch.log 2>&1 & echo PID:$!
    ping_count: 3
    ping_timeout_sec: 2.0
    ssh_connect_timeout_sec: 10.0

  ekf:
    topic: "/amr/ekf/odom"
    window_size: 20            # samples in the sliding-window
    velocity_threshold_mps: 0.05
    timeout_sec: 120.0

  optitrack:
    topic: "/optitrack/rigid_body"
    expected_frame_id: "drone"
    max_stamp_age_sec: 1.0     # message must be this fresh
    check_timeout_sec: 30.0
    retry_delay_sec: 3.0       # wait after auto-launching client

  drone:
    state_topic: "/drone/state"
    camera_topic: "/camera/image_raw"
    battery_topic: "/battery_state"
    cmd_vel_topic: "/cmd_vel"
    battery_min_pct: 20.0
    camera_timeout_sec: 30.0
    battery_timeout_sec: 10.0
    state1_timeout_sec: 60.0   # per-state transition deadlines
    state2_timeout_sec: 300.0
    state3_timeout_sec: 30.0
    state4_timeout_sec: 60.0
    video_filename_topic: "/drone/video_filename"
    telemetry_filename_topic: "/drone/telemetry_filename"

  video:
    file_appear_timeout_sec: 180.0   # deadline for video topics after landing

  aruco:
    amr_marker_id: 0
    goal_marker_id: 1
    amr_pose_topic: "/aruco/amr/pose"
    goal_pose_topic: "/aruco/goal/pose"

  marker_localizer:
    workspace_path: "/CHANGEME/path/to/workspace"   # ← must be set
    service_name: "/localize_markers"
    service_timeout_sec: 300.0
    server_ready_timeout_sec: 60.0

  map_builder:
    background_image_path: "/CHANGEME/path/to/background.png"  # ← must be set
    action_name: "build_arena_map"
    action_timeout_sec: 300.0
    server_ready_timeout_sec: 30.0
    drone_map_topic: "/drone/map"

  logging:
    log_dir: "/tmp/mission_orchestrator_logs"
    log_level: "INFO"    # DEBUG | INFO | WARNING | ERROR
```

**Before first run**, replace the two `CHANGEME` entries:

| Key | Expected value |
|---|---|
| `marker_localizer.workspace_path` | Absolute path to your ROS 2 workspace root (the one that contains `arena_marker_localizer`) |
| `map_builder.background_image_path` | Absolute path to the arena background PNG used by `arena_map_builder` |

---

## Build

```bash
cd <workspace_root>
colcon build --symlink-install \
  --packages-select \
    arena_marker_localizer_interfaces \
    arena_marker_localizer \
    arena_map_builder_msgs \
    arena_map_builder \
    mission_orchestrator
source install/setup.bash
```

`python3-paramiko` must be installed in the active Python environment:

```bash
pip install paramiko   # or: sudo apt install python3-paramiko
```

---

## Running

```bash
ros2 launch mission_orchestrator orchestrator.launch.py \
  config_file:=/absolute/path/to/orchestrator_params.yaml
```

If `config_file` is omitted the bundled `config/orchestrator_params.yaml`
is used (the one with `CHANGEME` placeholders — fill those in first).

The node prints a structured log to stdout and to a timestamped file under
`logging.log_dir`.  Use `log_level: "DEBUG"` to see every stage detail.

### Stopping

`Ctrl+C` triggers the abort sequence: drone velocity is zeroed,
`tello_map` is killed (SIGINT → 5 s → SIGKILL), then `tello_driver`,
then all other subprocesses, and finally the SSH connection and remote AMR
process are cleaned up.

---

## Abort behaviour

Any stage can raise `MissionAbortError`.  When that happens, `run()`
catches it and calls `_abort()`, which:

1. Publishes zero `cmd_vel` (10 × 50 ms) to halt the drone.
2. Sends SIGINT to `tello_map`; waits 5 s; sends SIGKILL if needed.
3. Waits 5 s then kills `tello_driver`.
4. Kills all other managed subprocesses (SIGINT → SIGKILL).
5. Sends `kill -INT <pid>` to the remote AMR process over SSH, then
   `kill -9` after 3 s.
6. Closes the SSH connection.

Abort is idempotent — calling it twice is safe.

---

## Testing

The test suite lives in `test/` and uses **pytest** with in-process ROS 2
mock nodes.  No hardware and no external processes are needed.

### Structure

```
test/
├── conftest.py          # shared fixtures + helpers
├── mocks/
│   ├── rasp_mock.py     # publishes /amr/ekf/odom
│   ├── optitrack_mock.py # publishes /optitrack/rigid_body
│   └── drone_mock.py    # full drone state machine + video topics
├── test_rasp.py         # stages 01-04 (Raspberry Pi + EKF)
├── test_optitrack.py    # stage 05 (OptiTrack)
└── test_drone.py        # stages 06-09 (Tello preflight + states)
```

### How each test works

Each test file defines a `_OrchestratorUnderTest` subclass that **no-ops**
every stage except the ones under test.  A `TestableOrchestratorNode`
base class overrides `_abort()` to record the call without killing
real processes.  Mock ROS 2 nodes are spun in a shared `MultiThreadedExecutor`.

`node.run()` is called on the main thread (blocking).  Assertions check
`node._mission_complete` (happy path) or `node.abort_called` (abort path).

### Running the tests

```bash
cd <workspace_root>/src/mission_orchestrator
python3 -m pytest test/ -v
```

> `colcon test` uses the `setup.py test` runner which invokes unittest and
> does not understand pytest fixtures — run pytest directly as shown above.

### Test coverage

| File | Scenarios covered |
|---|---|
| `test_rasp.py` | Happy path; ping fails; SSH auth fails; EKF timeout; EKF stays unstable |
| `test_optitrack.py` | Happy path; no message; wrong `frame_id`; stale stamp |
| `test_drone.py` | Happy path; battery low; camera timeout; state 1/2/3 timeout |

### Mock nodes

**`RaspMockNode`** — publishes `nav_msgs/Odometry` on `/amr/ekf/odom` at
10 Hz.  Pass `velocity_mps=0.0` for a stable EKF or a value above the
threshold to simulate an unstable one.

**`OptiTrackMockNode`** — publishes `geometry_msgs/PoseStamped` on
`/optitrack/rigid_body` at 10 Hz.  Parameters:
- `frame_id` — set to something other than `"drone"` to trigger the
  frame\_id mismatch abort.
- `active=False` — publishes nothing (triggers the no-message timeout).
- `stamp_offset_sec` — large negative value (e.g. `−5.0`) makes the
  stamp appear stale.

**`DroneMockNode`** — simulates the full Tello interface:
- Publishes camera frames and battery state from creation.
- Calls `start_mission()` (triggered by the overridden stage 08) to
  start the state machine thread: publishes states 0 → 1 → 2 → 3 → 4.
- After state 4, copies `scan.mp4` and `telemetry.csv` from the bundled
  `arena_map_builder/data/drone_scans/scan10/` dataset into `video_dir`
  and publishes their paths on the video topics.
- `stuck_at_state=N` freezes the machine at state N to trigger a timeout.
- `camera_active=False` / `battery_pct` control preflight failures.

---

## Architecture notes

- **Threading model**: `MultiThreadedExecutor` spins in a background
  thread.  `run()` blocks the main thread and calls stages sequentially.
  All "wait for a topic" synchronisation uses `threading.Event.wait()`.

- **Config at runtime**: the YAML is loaded by `_load_config()` at node
  init; no ROS parameters are declared — this keeps the node simple and
  avoids the parameter namespace complexity of large configs.

- **Process lifecycle**: every `subprocess.Popen` handle is stored in
  `self._processes` (keyed by name).  `_abort()` iterates the dict and
  calls `_kill_proc()` on each, which sends SIGINT, waits, then SIGKILL.

- **Service calls from the main thread**: `client.call_async()` is used
  with a `threading.Event` done-callback so the main thread can block
  without conflicting with the executor spinning in the background.

- **Action client**: same pattern — `send_goal_async` → event-wait →
  `get_result_async` → event-wait.
