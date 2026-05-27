# LiDAR mapping - apt packages only

```bash
colcon build --symlink-install --packages-select amr_optitrack arena_map_builder_msgs  arena_marker_localizer_interfaces local_costmap trajectory_planner arena_map_builder  arena_marker_localizer map_fusion oradar_lidar optitrack_client tello_driver tello_pos_control tello_msgs
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

ros2 launch amr_bringup launch_rasp.py
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
ros2 launch arena_marker_localizer marker_localizer.launch.py params_file:=$WS_YAML

# Terminal 2
source install/setup.sh

ros2 service call /localize_markers arena_marker_localizer_interfaces/srv/LocalizeMarkers \
  "{video_path: '/absolute/path/to/your/video.mp4',
    optitrack_csv: '/absolute/path/to/your/optitrack.csv'}"
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
python3 src/mission_orchestrator/scripts/run_hw_test_s10_s20.py \
    --file-path /home/jetson/collab_nav_ground-jetson/src/arena_map_builder/data/drone_scans/scan10 \
    --aruco-ids '[1, 3]' \
    --touch-files \
    --trajectory-planner=false
```