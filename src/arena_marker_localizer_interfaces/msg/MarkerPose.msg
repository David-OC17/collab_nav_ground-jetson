# One ArUco marker's aggregated pose in the arena map frame.
#
# Fields
# ──────
#   id                    : ArUco marker integer ID (unique inside the dictionary used)
#   pose_3d               : Full 3D pose in the map frame (metres + quaternion)
#   pose_2d               : Same pose flattened onto the floor plane (x,y in metres, theta=yaw)
#   pose_with_covariance  : PoseWithCovarianceStamped ready for slam_toolbox / map_fusion.
#                           Diagonal covariance: xx=yy=base_var_pos/n_obs, yaw=base_var_yaw/n_obs.
#                           Stamp is the service-call time; frame_id matches the map frame.
#   cell_x, cell_y        : Index into the OccupancyGrid this map was built for
#                           (uint32; clamped to grid bounds upstream)
#   n_observations        : How many video frames contributed to this aggregate
#                           (after outlier rejection)

uint32                                   id
geometry_msgs/Pose                       pose_3d
geometry_msgs/Pose2D                     pose_2d
geometry_msgs/PoseWithCovarianceStamped  pose_with_covariance
uint32                                   cell_x
uint32                                   cell_y
uint32                                   n_observations
