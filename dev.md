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

ros2 launch oradar_ros ms200_scan.launch.py

ros2 run tf2_ros static_transform_publisher \
  --x 0.07 --y 0.0 --z 0.13 \
  --roll 0 --pitch 0 --yaw 0 \
  --frame-id base_link \
  --child-frame-id lidar

ros2 run odom_to_tf odomTF_node

ros2 run topic_tools relay /ekf/odom /odom

ros2 launch slam_gmapping slam_gmapping.launch.py use_sim_time:=false

rviz2
```

In Rasp:
```bash
ros2 launch ekf_amr ekf_launch.py
ros2 launch amr_bringup position_control_launch.py
ros2  launch network_bridge tcp.launch.py
```