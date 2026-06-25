# collab_nav_ground — Jetson Workspace

Drone-guided autonomous navigation: a collaborative **UAV–AMR** system for
GPS-denied cooperative localization and navigation. A DJI Tello surveys a 4 × 4 m
arena from above to build a global map and locate ArUco goal markers, then hands
the result to a ground robot (Yahboom RDK X3 + Jetson Orin Nano) which fuses it
with onboard LiDAR to plan and drive a collision-free path.

This repository is the **Jetson Orin Nano** ROS 2 Humble workspace. The companion
RDK X3 workspace (`collab_nav_ground-rasp`) hosts the EKF and hardware bringup.

For the full architecture, pipeline, and design rationale see **[DOCS.md](DOCS.md)**.

## Package map

The whole mission is driven by `mission_orchestrator`, a 20-stage sequencer that
launches the packages below as subprocesses.

| Package | Role | Launched by orchestrator |
|---|---|:--:|
| `mission_orchestrator` | Mission state machine / overseer | — (entry point) |
| `arena_map_builder` (+ `_msgs`) | Aerial image stitching → occupancy grid | ✅ |
| `arena_marker_localizer` (+ `_interfaces`) | ArUco PnP pose estimation (service) | ✅ |
| `aruco_localizer` | Lightweight ArUco detection (AMR camera) | ✅ |
| `world_mapper` | Running ground occupancy map from LiDAR | ✅ |
| `fusion` | ICP fusion of aerial + LiDAR maps | ✅ |
| `trajectory_planner` | A* path planning + spline follower | ✅ |
| `emergency_stop` | Automatic Emergency Braking (AEB) | ✅ |
| `frontier_explorer` | Fallback frontier exploration | ✅ (fallback) |
| `amr_drone_nav` | World↔odom TF alignment (`alignment_node`) | ✅ |
| `amr_optitrack` | OptiTrack pose bridge | ❌ standalone teleop tool |

**Submodules** (`git submodule update --init --recursive`):

| Submodule | Role |
|---|---|
| `src/collab_nav_uav` | UAV subsystem (`tello_driver`, `tello_pos_control`, `tello_controller`, `tello_calibrator`, `tello_msgs`) |
| `src/optitrack_client` | VRPN → ROS OptiTrack client |
| `src/oradar_ros` | OraDAR MS200 LiDAR driver (builds `oradar_lidar`) |
| `src/LightGlue-ONNX-Jetson` | SuperPoint + LightGlue ONNX inference (GPU stitching) |
| `network_bridge` | SecureEnvelope security middleware |

## Quickstart

```bash
# 1. Fetch submodules and Git LFS data (scans, recorded maps, videos)
git submodule update --init --recursive
git lfs pull

# 2. Build (workspace uses colcon + ament)
colcon build --symlink-install
source install/setup.bash

# 3. Run the full mission (fill in CHANGEME values in the YAML first)
ros2 launch mission_orchestrator orchestrator.launch.py \
    config_file:=/abs/path/to/src/mission_orchestrator/config/orchestrator_params.yaml
```

Large datasets (`src/arena_map_builder/data`, `src/mission_orchestrator/recorded_data`)
are stored with **Git LFS** — install `git-lfs` before cloning.

## Documentation

- **[DOCS.md](DOCS.md)** — system architecture, end-to-end pipeline, design decisions.
- Per-package details live in `src/<package>/DOCS.md` (indexed in DOCS.md §8).

## Housekeeping

Build artifacts (`build/`, `install/`, `log/`) and the `vo/` and stitching
`sweep/` outputs are git-ignored. To reclaim local disk and rebuild cleanly:

```bash
rm -rf build install log && colcon build --symlink-install
```
