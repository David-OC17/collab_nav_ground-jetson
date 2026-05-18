# One ArUco marker's aggregated pose in the arena map frame.
#
# Fields
# ──────
#   id              : ArUco marker integer ID (unique inside the dictionary used)
#   pose_3d         : Full 3D pose in the map frame (metres + quaternion)
#   pose_2d         : Same pose flattened onto the floor plane (x,y in metres, theta=yaw)
#   cell_x, cell_y  : Index into the OccupancyGrid this map was built for
#                     (uint32; clamped to grid bounds upstream)
#   n_observations  : How many video frames contributed to this aggregate
#                     (after outlier rejection)

uint32                id
geometry_msgs/Pose    pose_3d
geometry_msgs/Pose2D  pose_2d
uint32                cell_x
uint32                cell_y
uint32                n_observations
