# map_fusion

A ROS 2 **Humble** package that fuses a **drone-generated global map** with an
**incrementally built SLAM Toolbox map** by estimating the rigid `SE(2)`
transform between their frames, then serves the aligned result to Nav2.

The drone sees the whole 4×4 m arena from above (filled obstacle blobs, frame
`world`); the AMR's SLAM map grows over time (lidar contours, frame
`slam_map`). The two frames are unrelated until this node estimates

```
T_world_slam = (tx, ty, theta)
```

and broadcasts it on TF, so Nav2 can plan on the reliable drone prior while the
SLAM map corrects it locally.

---

## Pipeline

| Stage | What it does | Where |
|-------|--------------|-------|
| 1 | Threshold + morphological-gradient edges of the drone map | `preprocessing.py` |
| 2 | Same for each incoming SLAM map | `preprocessing.py` |
| 3 | Decide coarse re-search vs. ICP warm-start | `map_fusion_node.py` |
| 4 | Coarse global `SE(2)` search (per-rotation FFT cross-correlation) | `coarse_search.py` |
| 5 | ICP refinement with per-iteration outlier rejection | `icp.py` |
| 6 | Validate: convergence, sanity delta, confidence gate | `map_fusion_node.py` |
| 7 | Broadcast TF, reproject the SLAM grid, publish confidence | `map_fusion_node.py` |

---

## Build

Drop the package into a workspace `src/` and build:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select map_fusion
source install/setup.bash
```

Python dependencies (`numpy`, `scipy`, `opencv`) are resolved by `rosdep`.

## Run the offline demo (no hardware)

```bash
ros2 launch map_fusion demo.launch.py
```

This starts `mock_arena` (synthetic drone/SLAM/ArUco publishers) and the fusion
node. `mock_arena` prints its **ground-truth** `T_world_slam` at startup; watch
`/fusion/status` and confirm the node converges close to it:

```bash
ros2 topic echo /fusion/status
ros2 topic echo /fusion/confidence
```

In RViz, set the fixed frame to `world` and add two `Map` displays —
`/drone/map` and `/fusion/slam_reprojected` — to watch the SLAM layer snap into
alignment as more of the arena is revealed.

## Run against real inputs

```bash
ros2 launch map_fusion map_fusion.launch.py
# or with a custom parameter file:
ros2 launch map_fusion map_fusion.launch.py params_file:=/path/to/params.yaml
```

Edit `config/map_fusion_params.yaml` to match your topics, frames and tuning.

## Nav2 integration

`config/nav2_params.yaml` is a **fragment**, not a full Nav2 config. Merge its
`global_costmap` block into your own `nav2_params.yaml`. It stacks two
`StaticLayer`s — the drone map and `/fusion/slam_reprojected` — plus an
inflation layer sized to absorb residual alignment error.

---

## Interfaces

**Subscribes**

| Topic | Type | Notes |
|-------|------|-------|
| `/drone/map` | `nav_msgs/OccupancyGrid` | global prior, frame `world`, latched |
| `/map` | `nav_msgs/OccupancyGrid` | SLAM Toolbox map, frame `slam_map`, latched |
| `/aruco/amr_pose` | `geometry_msgs/PoseWithCovarianceStamped` | AMR pose in `world` (translation seed) |

**Publishes**

| Topic | Type | Notes |
|-------|------|-------|
| `/fusion/slam_reprojected` | `nav_msgs/OccupancyGrid` | SLAM map reprojected into `world`, latched |
| `/fusion/confidence` | `std_msgs/Float32` | alignment confidence in [0, 1] |
| `/fusion/status` | `std_msgs/String` | human-readable status / warnings |
| TF `world → slam_map` | static transform | the estimated `T_world_slam` |

The node also *listens* to TF `slam_map → base_link` to turn the ArUco pose
into a proper translation seed (see below).

---

## Design decisions worth knowing

**Hand-written ICP, library nearest-neighbour.** SciPy/OpenCV/NumPy carry the
heavy lifting, but the ICP *loop* is explicit. The spec requires rejecting
correspondences beyond `2 × median` distance, re-evaluated every iteration —
off-the-shelf ICPs (Open3D included) do not expose that, so the loop is
hand-written while `scipy.spatial.cKDTree` still does the expensive
nearest-neighbour search. Open3D was therefore left out entirely, which also
keeps `rosdep` clean.

**FFT coarse search.** For each candidate rotation, the inner translation
search collapses from `O(N_t²)` to a single FFT cross-correlation
(`O(N log N)`). The score's mask term cancels: every SLAM edge cell comes from
an *occupied* cell, occupied cells are *known*, so `S' ⊆ M` and the denominator
reduces to the (constant) SLAM edge count.

**`edge_kernel_size` is a radius.** A literal 1×1 structuring element is the
identity and yields *no* edges, so the parameter is interpreted as a radius
(default 1 → 3×3 kernel).

**`coarse_translation_step_m` as peak spacing.** The FFT gives correlation at
full drone-grid resolution. This parameter is used as the minimum separation
between extracted peaks, so the top-K candidates are genuinely distinct seeds
rather than K adjacent cells.

**ArUco seed via TF.** The ArUco pose is the *AMR in `world`*, but the search
needs a seed for `T_world_slam`. The node computes
`T_world_slam ≈ T_world_amr ∘ inv(T_slam_amr)`, looking up `T_slam_amr` from
TF (`slam_map → base_link`). If that lookup fails it falls back to the raw
ArUco position — the search radius absorbs the error.

**Reprojection by splatting.** Each known SLAM cell is splatted as a small
square (`ceil(slam_res / out_res)` wide) to avoid holes when upsampling
0.05 → 0.02 m/cell. Output cells with no SLAM contribution stay `-1`, so the
unknown mask is preserved.

---

## Tests

```bash
colcon test --packages-select map_fusion
# or, the algorithm tests alone (pure numpy/scipy, no ROS runtime):
python -m pytest test/test_alignment.py -v
```

`test/test_alignment.py` covers transform composition, the Kabsch fit, ICP
recovery from a perturbed guess, the residual→confidence mapping, and the
coarse-search basin.

## Notes on the spec's open issues

* **Symmetry (#3)** — when the top two coarse candidates fall within
  `symmetry_score_tolerance`, a warning is published on `/fusion/status`.
  ICP still refines every candidate and the lowest-residual one wins.
* **Live ArUco tracking (#4)** — the `/aruco/amr_tracking` subscription exists
  behind the `use_live_tracking` flag; the callback is currently a stub where a
  direct, ICP-free estimator would slot in.
* **Initial heading (#1)** — the seed computation already yields a heading
  estimate; the coarse search still sweeps the full rotation range as the spec
  requires, but an IMU/TF heading prior could later narrow it.
