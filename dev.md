## LiDAR mapping - apt packages only

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
ros2 param set /build_arena_map_server transfer.background_path src/arena_map_builder/config/background.png

python3 src/arena_map_builder/arena_map_builder/example_client.py /abs/path/video.mp4
```

ONNX Runtime GPU: onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl

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
    --video  src/arena_map_builder/data/static_scans/video3/scan.mp4 \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/static_scan3.yaml \
    --csv    src/arena_map_builder/data/static_scans/video3/telemetry.csv \
    \
    --video  src/arena_map_builder/data/static_scans/video4/scan.mp4 \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/static_scan4.yaml \
    --csv    src/arena_map_builder/data/static_scans/video4/telemetry.csv \
    \
    --config     src/arena_marker_localizer/config/default.yaml \
    --intrinsics src/arena_marker_localizer/config/calibration.yaml \
    --verbose

# Check sync
src/arena_marker_localizer/scripts/check_sync --video /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/scan.mp4 --csv /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/telemetry.csv

# Calibrate bias
src/arena_marker_localizer/scripts/calibrate_bias \
    --video  src/arena_map_builder/data/drone_scans/scan18/scan.mp4 \
    --csv    src/arena_map_builder/data/drone_scans/scan18/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/scan18.yaml \
    \
    --config     src/arena_marker_localizer/config/default.yaml \
    --intrinsics src/arena_map_builder/data/drone_scans/scan18/calibration.yaml \
    --out        src/arena_marker_localizer/config/corrected_T_map_from_opti.yaml

# src/arena_marker_localizer/scripts/calibrate_bias_v2 \
src/arena_marker_localizer/scripts/calibrate_bias \
    --video  src/arena_map_builder/data/drone_scans/scan18/scan.mp4 \
    --csv    src/arena_map_builder/data/drone_scans/scan18/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/scan18.yaml \
    \
    --video  src/arena_map_builder/data/drone_scans/scan19/scan.mp4 \
    --csv    src/arena_map_builder/data/drone_scans/scan19/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/scan19.yaml \
    \
    --video  src/arena_map_builder/data/drone_scans/scan20/scan.mp4 \
    --csv    src/arena_map_builder/data/drone_scans/scan20/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/scan20.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan2/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan2/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan2.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan3/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan3/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan3.yaml \
    \
    --config     src/arena_marker_localizer/config/default.yaml \
    --intrinsics src/arena_map_builder/data/drone_scans/scan18/calibration.yaml \
    --out        src/arena_marker_localizer/config/corrected_T_map_from_opti.yaml
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
# Post-scan map pipeline, no AMR (map builder + localizer + mapping + planner)
python3 src/mission_orchestrator/scripts/run_hw_test_postscan.py \
    --file-path /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan10 \
    --aruco-ids '[1, 3]' \
    --touch-files \
    --trajectory-planner=false

# Post-scan map pipeline + AMR bringup
python3 src/mission_orchestrator/scripts/run_hw_test_postscan_amr.py \
    --file-path /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan10 \
    --aruco-ids '[0, 2]' \
    --touch-files \
    --trajectory-planner=false
```

Test planning and control loop that executes after orchestrator's handoff:
```bash
# Once per scan (slow — runs map builder + localizer):
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 10 --aruco-ids '[0, 2]'
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 15 --aruco-ids '[5, 0]'
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 16 --aruco-ids '[5, 3]'
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 17 --aruco-ids '[3, 2]'
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 18 --aruco-ids '[3, 2]'
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 20 --aruco-ids '[5, 0]' --ground-truth src/arena_marker_localizer/config/aruco_pose_gt/scan20.yaml
python3 src/mission_orchestrator/scripts/save_scan_data.py --scan-id 21 --aruco-ids '[5, 0]' --ground-truth src/arena_marker_localizer/config/aruco_pose_gt/scan21.yaml

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

## Making async optitrack-drone videos

```bash
ros2 run tello_pos_control record_pose
```

## Transform telemetry

```bash
transform_telemetry \
  --csv /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/static_scans/video1/telemetry.csv \
  --x 0.0 --y 0.0 --z 0.0 \
  --roll 0.0 --pitch 0.0 --yaw 1.570795
```

## Enable optitrack + EKF + Teleoperated AMR

```bash
home/jetson/collab_nav_ground-jetson/scripts/amr_teleop_optitrack.sh 
```

## Run planner with custom coordinates 
./scripts/custom_trajectory.sh 

Obstain starting with:
```bash
src/arena_marker_localizer/scripts/calibrate_bias_v3 \
    --video  src/arena_map_builder/data/manual_scans/scan2/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan2/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan2.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan3/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan3/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan3.yaml \
    \
    --config     src/arena_marker_localizer/config/default.yaml \
    --intrinsics src/arena_map_builder/data/manual_scans/scan1/calibration.yaml \
    --out        src/arena_marker_localizer/config/corrected_T_map_from_opti.yaml \
    --max-cam-offset 0.10
```
```bash
src/arena_marker_localizer/scripts/calibrate_bias_v3 \
    --test-video  src/arena_map_builder/data/drone_scans/scan18/scan.mp4 \
    --test-csv    src/arena_map_builder/data/drone_scans/scan18/telemetry.csv \
    --test-gt     src/arena_marker_localizer/config/aruco_pose_gt/scan18.yaml \
    \
    --test-video  src/arena_map_builder/data/drone_scans/scan19/scan.mp4 \
    --test-csv    src/arena_map_builder/data/drone_scans/scan19/telemetry.csv \
    --test-gt     src/arena_marker_localizer/config/aruco_pose_gt/scan19.yaml \
    \
    --test-video  src/arena_map_builder/data/drone_scans/scan20/scan.mp4 \
    --test-csv    src/arena_map_builder/data/drone_scans/scan20/telemetry.csv \
    --test-gt     src/arena_marker_localizer/config/aruco_pose_gt/scan20.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan2/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan2/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan3/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan3/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan4/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan4/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan5/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan5/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan6/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan6/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan8/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan8/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan9/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan9/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --video  src/arena_map_builder/data/manual_scans/scan10/scan.mp4 \
    --csv    src/arena_map_builder/data/manual_scans/scan10/telemetry.csv \
    --gt     src/arena_marker_localizer/config/aruco_pose_gt/manual_scan-config1.yaml \
    \
    --config     src/arena_marker_localizer/config/default.yaml \
    --intrinsics src/arena_marker_localizer/config/calibration.yaml \
    --out        src/arena_marker_localizer/config/corrected_T_map_from_opti.yaml \
    --max-cam-offset 0.10
```

## Isaac Ros Visual Slam

```bash
  docker start isaac_ros_vslam
  docker exec -it isaac_ros_vslam bash
  ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py
```

## XGBoost data preparation

```bash

# ── 0. Source ROS in every terminal you open ──────────────────────────────
source /home/jetson/collab_nav_ground-jetson/install/setup.bash
cd /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/sweep


# ── 1. Run the sweep (overnight) ─────────────────────────────────────────
# Terminal 1
ROS_DOMAIN_ID=1 python sweep.py \
    --video ../../data/drone_scans/scan18/scan.mp4 \
    --output-dir results/scan18 \
    --goal-marker-id 3 --amr-marker-id 2

# Terminal 2
ROS_DOMAIN_ID=2 python sweep.py \
    --video ../../data/drone_scans/scan21/scan.mp4 \
    --output-dir results/scan21 \
    --goal-marker-id 5 --amr-marker-id 0

# ── 2. Manual pass/fail/unsure labeling ───────────────────────────────────
python3 label.py \
    --results-dirs results \
    --labels-file  labels.yaml
# Controls:  p=PASS   f=FAIL   s=UNSURE   u=UNDO   q=QUIT
# Safe to quit and resume — labels are saved after every keypress.
# If you run more sweeps later, add their results dirs:
#   python3 label.py --results-dirs results results2 --labels-file labels.yaml


# ── 3. Extract features → CSV ─────────────────────────────────────────────
python3 extract_features.py \
    --results-dirs results \
    --background   /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/config/background.png \
    --labels-file  labels.yaml \
    --output       features.csv
# Because sweep.py now writes all 46 diagnostics into each metrics.yaml,
# this step reads them directly (no pipeline re-run) and finishes in seconds.
# Output: features.csv  — one row per run, ~57 columns ready for XGBoost.
```

Major stages and substages:
01. Optitrack bringup
    01.a.  Check /optitrack/rigid_body presence + header; launch client if absent
    01.b.  Do sanity check of /optitrack/rigid_body contents
02. Arena map builder bringup
    02.a.  Configure background_path parameter
    02.b.  Configure online/offline mode (default online)
    (Async) 02.c. Server will respond with OccupancyGrid and ego publishes
03. Drone routine
    03.a.  Connect to Tello WiFi (nmcli scan + connect on wlx14ebb67dae0b)
    03.b.  Launch tello_driver
    03.c.  Drone preflight: verify /camera/image_raw live, /battery_state ≥ min %
    03.d.  Launch tello_map (drone takes off and executes scanning routine)
    03.e.  (If online) Send start request to map builder (to consume incoming drone /camera/image_proc)
    03.f.  Observe drone state transitions 1→2→3→4 with per-stage timeouts
    03.g.  Wait for scan.mp4 and telemetry.csv to appear in the configured video dir, fresh within max_age_sec
    03.h.  Verify scan.mp4 integrity via ffmpeg
04. Launch Aruco localizer server
    04.a. Call service with new scan.mp4
    04.b. Collect response from Aruco localizer server and stitching pipeline and publish to /aruco/.../pose (built from map-builder marker position + aruco-localizer orientation)
05. Isaac ROS Visual SLAM (cuSLAM) bringup
    05.a. Verify Intel Realsense D435i is plugged in
    05.b. Start Docker container and enter
    05.c. Launch visual SLAM for Realsense camera
    05.d. Check /visual_slam/tracking/odometry has a valid output (from outside the container)
06. Rasp bringup
    06.a.  Ping Raspberry Pi
    06.b.  SSH connect to Raspberry Pi
    06.c.  Start amr_bringup systemd service on Raspberry Pi and verify active
    06.d.  Wait for /imu/data_raw to publish 200 messages (IMU running at 100 Hz)
07. Emergency stop bringup
    07.a. Verify /amr/emergency_stop flag is innactive originally
08. Mapping bringup
    08.a.  Launch oradar lidar
    08.b.  Launch alignment_node (translates the /aruco/amr/pose to world->odom tf)
    08.c.  Launch odom-based mapper (no SLAM)
09. Trajectory planner bringup
10. Enter observer mode and log updates

TODO:
- Verify safe mode is active
- Verify stitching output is good, else fallback to frontier exploration


## Running in your laptop

```bash
mamba create -n arena_builder -c robostack-staging -c conda-forge \
    python=3.11 \
    ros-humble-desktop ros-humble-cv-bridge \
    ros-humble-nav-msgs ros-humble-geometry-msgs ros-humble-sensor-msgs \
    ros-humble-std-srvs ros-humble-action-msgs ros-humble-std-msgs \
    ros-humble-rosidl-default-generators \
    ros-humble-ament-cmake ros-humble-ament-cmake-python \
    colcon-common-extensions rosdep \
    'numpy<2' opencv pyyaml \
    compilers cmake pkg-config make ninja

mamba activate arena_builder

cd /home/david/Documents/UNI_S.8/Robo/project/collab_nav_ground-jetson
colcon build --packages-select arena_map_builder_msgs arena_map_builder \
    --symlink-install

source install/setup.bash

mamba install -c conda-forge 'cuda-version=12.*' cudnn
pip install onnxruntime-gpu
```
