# arena_marker_localizer

ROS 2 (Humble) service that locates ArUco markers in the arena map
frame by fusing:

* a drone aerial **video** (already undistorted with the calibrated intrinsics),
* the matching **OptiTrack pose log** (one CSV row per video frame),
* a known **camera-to-drone** static transform (6 numbers),
* a known **OptiTrack-to-map** static transform (6 numbers).

Markers seen multiple times across the video are aggregated with a
per-axis MAD outlier gate followed by a geometric-median fit.

Pipeline (per frame N)
----------------------

1. Read video frame N and CSV row N (strict 1:1 by row index).
2. Quality filter — drop blurry / codec-artifact frames using the same
   Laplacian-variance + DCT-grid scores as the stitcher.
3. Detect markers — every configured dictionary is scanned on every frame.
4. `solvePnP` per marker (IPPE_SQUARE) → pose in the camera frame.
5. Compose the static chain
   `T_map_from_opti  ·  T_opti_from_drone(t)  ·  T_drone_from_cam  ·  T_cam_from_marker`
   to get the marker pose in the map frame.
6. Per-detection sanity gate on `solvePnP` reprojection error.
7. After the loop, per-marker aggregation:
   - per-axis MAD gate on `(x, y, z, yaw)` — an outlier in any axis
     rejects the whole observation;
   - geometric median (Weiszfeld) of the surviving positions;
   - circular median of the surviving yaws.

Service interface
-----------------

`arena_marker_localizer/srv/LocalizeMarkers`:

```
string video_path
string optitrack_csv
---
bool                                  success
string                                message
arena_marker_localizer/MarkerPose[]   markers
```

`arena_marker_localizer/msg/MarkerPose`:

```
uint32                id
geometry_msgs/Pose    pose_3d
geometry_msgs/Pose2D  pose_2d
uint32                cell_x
uint32                cell_y
uint32                n_observations
```

`pose_3d` and `pose_2d` carry the same information; the 2D variant is
just a convenience for downstream consumers that don't want to unpack a
quaternion. Cell indices are computed from the configured grid
resolution and size and clamped to the grid bounds.

Code layout
-----------

```
arena_marker_localizer/
├── msg/MarkerPose.msg
├── srv/LocalizeMarkers.srv
├── arena_marker_localizer/      # pure-Python core (no rclpy)
│   ├── intrinsics.py            # OpenCV-YAML calibration reader
│   ├── quality.py               # blur + DCT-artifact gates
│   ├── optitrack.py             # CSV reader
│   ├── transforms.py            # static chain + Euler/quat helpers
│   ├── marker_detection.py      # multi-dictionary ArUco + solvePnP
│   ├── aggregation.py           # MAD gate + geometric median
│   ├── pipeline.py              # end-to-end driver
│   └── service_node.py          # ROS-side wrapper
├── config/default.yaml
├── launch/marker_localizer.launch.py
├── scripts/marker_localizer_service
├── example_client.py
├── CMakeLists.txt
├── package.xml
└── setup.py / setup.cfg
```

The pure-Python core is testable without rclpy:

```python
from arena_marker_localizer.pipeline import PipelineConfig, run_pipeline
from arena_marker_localizer.marker_detection import DictionaryConfig
from arena_marker_localizer.transforms import StaticTransform6DoF

cfg = PipelineConfig(
    intrinsics_path="/abs/path/calib.yaml",
    dictionaries=[
        DictionaryConfig(name="DICT_4X4_50", marker_size_m=0.10),
    ],
    T_drone_from_cam=StaticTransform6DoF(z=-0.05, pitch=3.14159/2),
    T_map_from_opti=StaticTransform6DoF(x=2.0, y=2.0),
    verbose=True,
)
results = run_pipeline("/abs/path/video.mp4", "/abs/path/opti.csv", cfg)
for marker_id, r in results.items():
    print(marker_id, r.position_m, r.yaw_rad, r.n_observations)
```

Build
-----

```bash
cd ~/ros2_ws/src
ln -s /path/to/arena_marker_localizer .
cd ~/ros2_ws
colcon build --packages-select arena_marker_localizer --symlink-install
source install/setup.bash
```

Run
---

```bash
ros2 launch arena_marker_localizer marker_localizer.launch.py
ros2 param set /marker_localizer_service intrinsics_path /abs/path/calib.yaml

# Either via the example client...
python3 example_client.py /abs/path/video.mp4 /abs/path/opti.csv

# ...or with the ros2 service CLI:
ros2 service call /localize_markers \
    arena_marker_localizer/srv/LocalizeMarkers \
    "{video_path: '/abs/path/video.mp4', optitrack_csv: '/abs/path/opti.csv'}"
```

Static-transform tips
---------------------

The two transforms in `T_drone_from_cam` and `T_map_from_opti` are the
only thing you usually need to tune.

**`T_drone_from_cam`** — typical for a downward-facing camera on a
quadrotor:
* `x = y = 0`, `z = -0.05` (camera sits 5 cm below the OptiTrack rigid-body origin).
* The OpenCV camera frame is `+X right, +Y down, +Z forward`. A downward camera
  with image up-direction aligned with the drone forward direction needs roughly
  `roll=pi`, `pitch=0`, `yaw=0` (flip Y), then small mounting-error tweaks.

**`T_map_from_opti`** — usually a pure translation if your OptiTrack X/Y
axes already align with the map's nav2 convention. If the arena is
4 m × 4 m and the OptiTrack origin is the arena centre, set
`x = 2.0, y = 2.0` (the OccupancyGrid origin is the bottom-left).
If your OptiTrack X points opposite to map X, flip with `optitrack.x_dir: -1`.
