# arena_map_builder

ROS 2 (Humble) action server that turns an aerial video of a robotics
arena into a probabilistic nav2 `OccupancyGrid`.

The package wraps two existing standalone pipelines:

* `drone_map_grid_gen.py` — video → top-down stitched map
* `transfer_obstacles.py` — noisy stitched map → clean obstacle map

and adds:

* a 3-pass consistency estimator (bbox vs grid projection + parameter
  perturbation) that produces a per-obstacle confidence in `[0, 1]`,
* an occupancy rasterizer with confidence-weighted exponential
  thickening — low-confidence obstacles get a larger, softer halo in
  the final grid.

Code layout
-----------

```
arena_map_builder/
├── action/
│   └── BuildArenaMap.action
├── arena_map_builder/                 # ROS-side Python code
│   ├── build_arena_map_server.py      # the action server node
│   ├── consistency.py                 # 3-pass consistency proxies
│   ├── occupancy.py                   # image → OccupancyGrid
│   └── processing/                    # VENDORED — do NOT edit here
│       ├── drone_map_grid_gen.py
│       └── transfer_obstacles.py
├── config/default.yaml                # all parameters with defaults
├── launch/build_arena_map.launch.py
├── scripts/build_arena_map_server     # console launcher
├── example_client.py
├── CMakeLists.txt
├── package.xml
└── setup.py / setup.cfg
```

The vendored processing modules under `arena_map_builder/processing/`
are exact copies of the standalone scripts and the ROS code never
modifies them. To upgrade the pipeline, drop in the new versions of
those two files and rebuild.

Build
-----

```bash
cd ~/ros2_ws/src
ln -s /path/to/arena_map_builder .
cd ~/ros2_ws
colcon build --packages-select arena_map_builder --symlink-install
source install/setup.bash
```

Run the server
--------------

```bash
ros2 launch arena_map_builder build_arena_map.launch.py
```

Set the background template path (required), then send a goal:

```bash
ros2 param set /build_arena_map_server transfer.background_path \
    /abs/path/to/background.png

# Either via the example client...
python3 example_client.py /abs/path/to/flight.mp4

# ...or directly via the CLI:
ros2 action send_goal /build_arena_map \
    arena_map_builder/action/BuildArenaMap \
    "{video_path: '/abs/path/to/flight.mp4'}" --feedback
```

Action interface
----------------

```
# BuildArenaMap.action
string video_path
---
nav_msgs/OccupancyGrid map
bool                  success
string                message
uint32                n_obstacles
float32               mean_consistency
string                debug_dir
---
string  stage      # stitching | transferring | consistency | occupancy | done
float32 progress   # 0..1 within the current stage
string  message
```

Parameters
----------

All tuning lives in `config/default.yaml`. The most important ones:

| Group        | Param                            | Default      |
|--------------|----------------------------------|--------------|
| `transfer`   | `background_path` (REQUIRED)     | `""`         |
| `transfer`   | `close_iterations`               | `3`          |
| `transfer`   | `use_florence2`                  | `false`      |
| `consistency`| `position_tol_cells`             | `0.5`        |
| `consistency`| `perturb_stab_tol`               | `0.25`       |
| `occupancy`  | `resolution_m_per_cell`          | `0.05`       |
| `occupancy`  | `arena_width_m / arena_height_m` | `3.9 / 3.9`  |
| `occupancy`  | `base_thickness_px`              | `8`          |
| `occupancy`  | `decay_rate`                     | `3.0`        |

Concurrency
-----------

A new goal arriving while one is running pre-empts the running one
(`prev.abort()`) before starting; this matches the
"allow cancellation mid-run" semantics requested.

Debug outputs
-------------

The action server never publishes intermediate images to ROS topics.
Instead, every major stage writes a PNG into `${debug_dir}`
(default `/tmp/arena_map_builder`), along with a `consistency.csv`
table of per-obstacle scores.
