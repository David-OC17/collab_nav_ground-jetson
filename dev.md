# LiDAR mapping - apt packages only

Deviation LiDAR center to robot center (geometric)
x - red: 7 cm
y - green: 0 cm
z - blue: 13 cm

```bash
colcon build --symlink-install --packages-select oradar_ros network_bridge yahboomcar_laser rf2o_laser_odometry odom_to_tf fixed_stamp_scan
```

```bash
ros2 launch oradar_ros ms200_scan.launch.py

ros2 run tf2_ros static_transform_publisher \
  --x 0.07 --y 0.0 --z 0.13 \
  --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link \
  --child-frame-id lidar

source ~/collab_nav_ground-jetson/install/setup.bash
ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py \
  laser_scan_topic:=/scan \
  base_frame_id:=base_link \
  odom_frame_id:=odom \
  publish_tf:=true

ros2 launch slam_toolbox online_async_launch.py \
  params_file:=/home/jetson/collab_nav_ground-jetson/config/slam_params.yaml \
  use_sim_time:=false

ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/cmd_vel

rviz2
```
