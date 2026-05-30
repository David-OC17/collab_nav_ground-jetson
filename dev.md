# LiDAR mapping - apt packages only

```bash
colcon build --symlink-install --packages-select amr_optitrack arena_map_builder_msgs  arena_marker_localizer_interfaces local_costmap trajectory_planner arena_map_builder  arena_marker_localizer map_fusion oradar_lidar optitrack_client tello_driver tello_pos_control tello_msgs emergency_stop
```

In Jetson:

```bash
ros2 launch oradar_lidar ms200_scan.launch.py

ros2 launch slam_toolbox online_async_launch.py

rviz2
```

In Rasp:

```bash
ros2 launch ekf_amr ekf_launch.py
ros2 launch amr_bringup position_control_launch.py

ros2 launch amr_bringup rasp_launch.py
```

## Arena map builder

To test using a mamba environment in an Ubuntu 22.04 install:

```bash
mamba create -n collab_nav -c robostack-staging -c conda-forge \
    python=3.11 ros-humble-desktop \
    ros-humble-nav-msgs ros-humble-geometry-msgs ros-humble-action-msgs \
    ros-humble-rosidl-default-generators ros-humble-ament-cmake \
    colcon-common-extensions catkin_tools rosdep \
    compilers cmake pkg-config make ninja
mamba activate arena

pip install opencv-python numpy

# Florence-2 only if needed:
pip install transformers torch pillow einops timm
```

If the system has a ROS2 installation we must unset some variables for the build to take those internal to the mamba environment:

```bash
# Verify
echo $AMENT_PREFIX_PATH

# Unset to use env internal
unset AMENT_PREFIX_PATH           
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH
unset ROS_DISTRO
unset ROS_VERSION
```

Execute a test via:

```bash
# Terminal 1
ros2 run arena_map_builder build_arena_map_server

# Terminal 2
ros2 param set /build_arena_map_server transfer.background_path /abs/path/background.png

python3 src/arena_map_builder/arena_map_builder/example_client.py /abs/path/video.mp4
```

## Arena marker localizer   

```bash
# Terminal 1
source install/setup.sh

# Pass yaml file explicitly, currently not finding default.yaml on its own
WS_YAML=$(find /path/to/workspace -name "default.yaml" -path "*arena_marker_localizer*" -not -path "*build*" | head -1)
echo "Using YAML: $WS_YAML"


# Terminal 2
source install/setup.sh

ros2 service call /localize_markers arena_marker_localizer_interfaces/srv/LocalizeMarkers \
  "{video_path: '/absolute/path/to/your/video.mp4',
    optitrack_csv: '/absolute/path/to/your/optitrack.csv'}"
```

Scripts:
```bash
# Calibrate extrinsincs
src/arena_marker_localizer/scripts/calibrate_extrinsics \
  --video /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/scan.mp4 \
  --csv   /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/telemetry.csv \
  --gt    /home/jetson/collab_nav_ground-jetson/src/arena_marker_localizer/config/calib_gt.yaml \
  --config     /home/jetson/collab_nav_ground-jetson/src/arena_marker_localizer/config/default.yaml \
  --intrinsics /home/jetson/collab_nav_ground-jetson/src/arena_marker_localizer/config/calibration.yaml

# Check sync
src/arena_marker_localizer/scripts/check_sync --video /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/scan.mp4 --csv /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/telemetry.csv

# Calibrate bias (can use multiple videos)
calibrate_bias \
  --video scan10.mp4 --csv scan10.csv \
  --gt    ground_truth.yaml \
  --config     src/arena_marker_localizer/config/default.yaml \
  --intrinsics src/arena_marker_localizer/config/calibration.yaml \
  --out   corrected_T_map_from_opti.yaml
```

## Emergency stop AMR 

```bash
ros2 launch amr_safety emergency_stop.launch.py
```

## Drone init

(Optional) In case OptiTrack is not yet available/up:
```bash
ros2 run optitrack_client optitrack_client
```

Verify Tello's Wifi connection availability and connect:
```bash
nmcli device wifi list ifname wlx14ebb67dae0b
sudo nmcli device wifi connect "TELLO-594992" ifname wlx14ebb67dae0b
```

```bash
ros2 launch tello_driver tello_driver.launch.py
ros2 launch tello_pos_control tello_map.launch.py
```

## Killing nodes

Using the symlink to the shell script:
```bash
# Configure
sudo ln -s /home/jetson/collab_nav_ground-jetson/scripts/kill_ros2_nodes.sh \
           /usr/local/bin/kill_ros2_nodes

# Kill everything in the default TARGETS list (local + RPi)
kill_ros2_nodes

# Kill a single node (checked on both Jetson and RPi)
kill_ros2_nodes --node driver_node

# Kill several specific nodes
kill_ros2_nodes --node driver_node --node ekf_node --node odometry_node

# Both --node syntaxes work
kill_ros2_nodes --node=driver_node --node=ekf_node

# Skip the RPi entirely (local Jetson only)
kill_ros2_nodes --no-rasp

# Local only, specific node
kill_ros2_nodes --no-rasp --node orchestrator_node

# Override retry/timing behaviour via env vars
MAX_RETRIES=10 kill_ros2_nodes --node driver_node
TERM_GRACE_SEC=5 KILL_RETRY_SEC=2 kill_ros2_nodes
MAX_RETRIES=3 TERM_GRACE_SEC=1 kill_ros2_nodes --node tello_driver_node --no-rasp

# Override RPi credentials (e.g. different deployment)
RASP_HOST=10.42.0.51 RASP_PASS=mypass kill_ros2_nodes --node driver_node

# Check exit code to know if everything was killed
kill_ros2_nodes --node driver_node
echo "Exit code: $?"   # 0 = all dead, 1 = something survived
```

## Testing orchestrator partial pipeline

```bash
# For scan10
python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20.py \
    --file-path /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan10 \
    --aruco-ids '[1, 3]' \
    --touch-files \
    --trajectory-planner=false

python3 src/mission_orchestrator/scripts/run_hw_test_s06_s20_amr.py \
    --file-path /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan10 \
    --aruco-ids '[0, 2]' \
    --touch-files \
    --trajectory-planner=false
```

Test planning and control loop that executes after orchestrator's handoff:
```bash
# Once per scan (slow — runs map builder + localizer):
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 --aruco-ids '[0, 2]'

# Many times (fast — only AMR bringup + planning):
python3 src/mission_orchestrator/scripts/run_hw_test_amr_nav.py --scan-id 10
```

## Benchmark LightGlue

See `src/LightGlue-ONNX/dev.md` for full setup instructions (environment, model export, troubleshooting).

```bash
cd src/LightGlue-ONNX

# SIFT baseline (~18 ms for 24 pairs)
uv run python benchmark_stitching.py --sift-only

# Unified SP+LG pipeline — Path A (re-runs SP per pair: ~6641 ms/frame, infeasible)
uv run python benchmark_stitching.py --provider cuda

# Split pipeline — Path B (extract once, match 24× with LG-only ONNX)
uv run python benchmark_stitching.py --split --provider cuda
uv run python benchmark_stitching.py --split --provider cpu
```

## Making static drone videos

```bash
nmcli device wifi list ifname wlx14ebb67dae0b
sudo nmcli device wifi connect "TELLO-594992" ifname wlx14ebb67dae0b

ros2 run optitrack_client optitrack_client

ros2 launch tello_driver tello_driver.launch.py
```

```bash
ros2 run tello_pos_control record_scan \
    --ros-args -p duration_sec:=120.0

# Custom output directory + undistortion (for calibration sessions)
ros2 run tello_pos_control record_scan --ros-args \
    -p output_dir:=/home/jetson/scans/calib01 \
    -p duration_sec:=60.0 \
    -p calibration_file:=/path/to/calibration.yaml

# Run until Ctrl+C
ros2 run tello_pos_control record_scan
```

## Transform telemetry

```bash
transform_telemetry \
  --csv /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/telemetry.csv \
  --x 0.0 --y 0.0 --z 0.0 \
  --roll 0.0 --pitch 0.0 --yaw 1.570795
```