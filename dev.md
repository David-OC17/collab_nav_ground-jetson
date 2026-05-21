# LiDAR mapping - apt packages only

Deviation LiDAR center to robot center (geometric)
x - red: 7 cm
y - green: 0 cm
z - blue: 13 cm

```bash
colcon build --symlink-install --packages-select oradar_ros network_bridge yahboomcar_laser rf2o_laser_odometry odom_to_tf fixed_stamp_scan slam_gmapping openslam_gmapping
```

In Jetson:

```bash
ros2 launch network_bridge tcp.launch.py

ros2 launch oradar_lidar ms200_scan.launch.py

ros2 run odom_to_tf odomTF_node

ros2 launch slam_gmapping slam_gmapping.launch.py use_sim_time:=false

rviz2
```

In Rasp:

```bash
ros2 launch ekf_amr ekf_launch.py
ros2 launch amr_bringup position_control_launch.py
ros2  launch network_bridge tcp.launch.py
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

python3 example_client.py /abs/path/video.mp4
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

ros2 param set /marker_localizer_service intrinsics_path \
 /absolute/path/to/your/calibration.yaml

ros2 service call /localize_markers arena_marker_localizer_interfaces/srv/LocalizeMarkers \
  "{video_path: '/absolute/path/to/your/video.mp4',
    optitrack_csv: '/absolute/path/to/your/optitrack.csv'}"
```