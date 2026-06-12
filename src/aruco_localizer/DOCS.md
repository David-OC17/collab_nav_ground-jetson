# `aruco_localizer` — ArUco-Based Absolute Pose Estimation

## 1. Role in the System

`aruco_localizer` provides **absolute 2D pose corrections** of the AMR in the `world` frame by detecting ArUco fiducial markers mounted at known positions on the arena walls. It is the primary absolute localization source for the ground robot throughout the navigation phase.

The node consumes the RGB stream of the Intel RealSense D435i mounted on the robot and publishes `geometry_msgs/PoseWithCovarianceStamped` on `/aruco_pose`. This topic is consumed by the `robot_localization` EKF running on the Yahboom RDK X3, which fuses it with wheel odometry and (optionally) VSLAM to produce the corrected pose estimate `/amr/ekf/odom`.

The node is launched in **Stage 08** of the mission orchestrator, after the AMR's EKF is running and the `world → odom` TF has been established by `alignment_node`. It runs continuously thereafter, providing absolute corrections that counteract the accumulation of dead-reckoning drift as the AMR navigates.

Placement in the localization chain:

```
RealSense D435i
    │  /camera/realsense2_camera/color/image_raw
    ▼
aruco_localizer
    │  /aruco_pose  (PoseWithCovarianceStamped, world frame)
    ▼
robot_localization EKF  (RDK X3)
    │  /amr/ekf/odom  (drift-corrected odometry)
    ▼
trajectory_planner / spline_follower
```

---

## 2. Detection and Pose Estimation Pipeline

Each image frame is processed through a five-step pipeline:

```
image → detect markers → PnP solve → TF composition → fusion → publish
```

### 2.1 Marker Detection

The detector uses OpenCV's ArUco module with the `DICT_4X4_50` dictionary (configurable via `aruco_dict_id`). The input image is converted to grayscale before detection. Detector parameters are tuned for the arena's artificial indoor lighting:

| Parameter | Value | Purpose |
|---|---|---|
| `adaptiveThreshWinSizeMin` | 3 | Smallest adaptive threshold window |
| `adaptiveThreshWinSizeMax` | 23 | Largest adaptive threshold window |
| `adaptiveThreshWinSizeStep` | 10 | Step size between window sizes |
| `minMarkerPerimeterRate` | 0.03 | Minimum marker size as a fraction of image perimeter |
| `maxMarkerPerimeterRate` | 4.0 | Maximum marker size |
| `polygonalApproxAccuracyRate` | 0.05 | Corner polygon approximation accuracy |
| `cornerRefinementMethod` | `CORNER_REFINE_SUBPIX` | Sub-pixel corner refinement |

After detection, two quality filters are applied before any marker is processed:

1. **ID whitelist:** Only markers whose IDs appear in `marker_ids` are processed. IDs absent from the list are silently ignored.
2. **Minimum perimeter:** Detections whose pixel perimeter is below `min_perimeter_px` are rejected. This eliminates blurry or distant detections where corner localization is unreliable.

### 2.2 Perspective-n-Point Solve

For each accepted detection, the transform $T_{\text{ar}}^{\text{cam}}$ (camera ← ArUco) is estimated from the four detected corners using `cv2.SOLVEPNP_IPPE_SQUARE`. This flag invokes the **Infinitesimal Plane-based Pose Estimation** closed-form solver, which is purpose-built for square planar targets and avoids the initialization sensitivity and local-minima risk of iterative PnP methods.

The 3D reference corners of the marker are defined in the marker's local frame, lying in the $z=0$ plane, centered at the origin:

$$
P_{\text{3D}} = \left\{
\begin{bmatrix}-s \\ +s \\ 0\end{bmatrix},\;
\begin{bmatrix}+s \\ +s \\ 0\end{bmatrix},\;
\begin{bmatrix}+s \\ -s \\ 0\end{bmatrix},\;
\begin{bmatrix}-s \\ -s \\ 0\end{bmatrix}
\right\}, \qquad s = \frac{\texttt{marker\_size}}{2}
$$

A distance filter is applied after the PnP solve: if $\|\mathbf{t}\| > \texttt{max\_marker\_dist}$, the detection is discarded. Long-range detections have amplified angular error because a given pixel uncertainty subtends a larger solid angle at distance.

### 2.3 Coordinate Frame Composition

The goal is $T_{\text{bf}}^{\text{world}}$ — the robot's `base_footprint` pose expressed in the `world` frame. Three transforms are combined:

$$
T_{\text{bf}}^{\text{world}} = T_{\text{ar}}^{\text{world}} \;\cdot\; \left(T_{\text{ar}}^{\text{cam}}\right)^{-1} \;\cdot\; \left(T_{\text{cam}}^{\text{bf}}\right)^{-1}
$$

where:

- $T_{\text{ar}}^{\text{world}}$ — the known pose of the marker in the global frame, retrieved from the TF tree (published as a static transform by the launch file).
- $T_{\text{ar}}^{\text{cam}}$ — the camera-to-marker transform produced by the PnP solve.
- $T_{\text{cam}}^{\text{bf}}$ — the extrinsic calibration from robot body to camera optical frame, looked up from the TF tree once at first detection and then cached for the node lifetime.

Caching $T_{\text{cam}}^{\text{bf}}$ is valid because this is a physically fixed transform on the robot. An explicit guard check ensures the TF is not cached until it is available — if the lookup fails (e.g., the static transform publisher has not started yet), the frame is dropped and the lookup is reattempted next frame.

### 2.4 Multi-Marker Fusion

When multiple markers are simultaneously visible, each yields an independent pose estimate $\{T_k, d_k\}$. These estimates are combined via **inverse-square-distance weighting**:

$$
w_k = \frac{1}{d_k^2 + \varepsilon}, \qquad \bar{w}_k = \frac{w_k}{\sum_j w_j}
$$

where $d_k = \|\mathbf{t}_k\|$ is the camera-to-marker distance from the PnP solve, and $\varepsilon = 10^{-6}$ guards against division by zero. Closer markers receive disproportionately higher weight because angular resolution — and therefore pose accuracy — is inversely proportional to range.

**Translation** is fused as a weighted linear combination:

$$
\mathbf{t}_{\text{fused}} = \sum_k \bar{w}_k \cdot \mathbf{t}_k
$$

**Rotation** is fused via linear quaternion averaging followed by renormalization. Before averaging, sign ambiguity is resolved by flipping any quaternion $\mathbf{q}_k$ whose dot product with the reference quaternion $\mathbf{q}_0$ is negative (ensuring all quaternions lie on the same hemisphere of $S^3$):

$$
\mathbf{q}_k \leftarrow \begin{cases} -\mathbf{q}_k & \text{if } \mathbf{q}_k \cdot \mathbf{q}_0 < 0 \\ \mathbf{q}_k & \text{otherwise} \end{cases}
\qquad
\mathbf{q}_{\text{fused}} = \frac{\sum_k \bar{w}_k \mathbf{q}_k}{\left\|\sum_k \bar{w}_k \mathbf{q}_k\right\|}
$$

This approximation is geometrically accurate when the individual estimates are close to each other — which holds here because all visible markers are in the same arena and, when the robot is stationary, yield consistent pose estimates.

---

## 3. Covariance Model

The EKF requires a covariance matrix to weight each incoming measurement against the process model. `aruco_localizer` uses a **distance-linear model** where uncertainty grows with the camera-to-marker range:

$$
\sigma_{xy} = \sigma_{xy,0} + \alpha_{xy} \cdot d
\qquad
\sigma_{\theta} = \sigma_{\theta,0} + \alpha_{\theta} \cdot d
$$

| Parameter | Default | Role |
|---|---|---|
| `cov_base_xy` | 0.01 m | Baseline position uncertainty at zero distance |
| `cov_dist_xy` | 0.02 m/m | Position uncertainty growth rate with distance |
| `cov_base_yaw` | 0.02 rad | Baseline yaw uncertainty at zero distance |
| `cov_dist_yaw` | 0.03 rad/m | Yaw uncertainty growth rate with distance |

The published covariance is a $6\times6$ diagonal in the order $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$. Only $\sigma_{xy}^2$ and $\sigma_{\theta}^2$ are assigned measured values; the remaining three diagonal entries are set to `HIGH_VAR = 9999.0`, signaling to the EKF that these dimensions carry no information and should not be corrected.

When multiple markers are fused, the reported covariance is the distance-weighted average of the per-marker covariances, computed alongside the pose average.

The recommended `robot_localization` EKF integration is:

```yaml
pose0: /aruco_pose
pose0_config: [true, true, false,    # x, y, z
               false, false, true,   # roll, pitch, yaw
               false, false, false, false, false, false,
               false, false, false]
pose0_differential: false
pose0_relative: false
```

---

## 4. Published Topics

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/aruco_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Default (depth 10) | Robot pose in `world` frame for the EKF |
| `/aruco_localizer/debug_image` | `sensor_msgs/Image` | Default (depth 10) | BGR frame annotated with detected marker outlines, coordinate axes, ID labels, and distance readout (only if `debug_image: true`) |
| `/aruco_localizer/detections_viz` | `visualization_msgs/MarkerArray` | Default (depth 10) | Arrow markers in RViz pointing in the estimated robot heading at the estimated position, one arrow per detected marker |

## 5. Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `image_topic` | `sensor_msgs/Image` | Color frames from the RealSense D435i (or alternative camera source) |
| `camera_info_topic` | `sensor_msgs/CameraInfo` | Camera intrinsics; subscribed continuously but only stored once |

The image and camera info topics are configurable, which allows the mission orchestrator to remap the localizer to the VSLAM RealSense IR stream when VSLAM is enabled (see §8).

## 6. TF Dependencies

| Transform | Source | Description |
|---|---|---|
| `world → aruco_<id>` | Static TF publisher (launch file) | Known pose of each marker in the global frame |
| `base_footprint → camera_color_optical_frame` | Static TF publisher (launch file) | Camera extrinsic calibration on the robot |

Both transforms are expected to be available in the TF buffer before the first image arrives. The camera extrinsic lookup is retried every frame until it succeeds, so a brief delay in the static publisher startup does not cause a hard failure.

---

## 7. Parameters

| Parameter | Default | Description |
|---|---|---|
| `marker_size` | 0.135 m | Physical side length of all markers (same for all IDs) |
| `aruco_dict_id` | 0 (`DICT_4X4_50`) | ArUco dictionary identifier |
| `marker_ids` | `[15, 16, 17, 21]` | IDs of markers whose world-frame poses are known |
| `world_frame` | `world` | Global reference frame |
| `base_frame` | `base_footprint` | Robot base frame |
| `camera_frame` | `camera_color_optical_frame` | Camera optical frame |
| `image_topic` | `/camera/realsense2_camera/color/image_raw` | Input image topic |
| `camera_info_topic` | `/camera/realsense2_camera/color/camera_info` | Input camera intrinsics topic |
| `pose_topic` | `/aruco_pose` | Output pose topic |
| `debug_image` | `true` | Publish annotated debug image |
| `publish_tf` | `false` | Broadcast `world → aruco_base_footprint_est` TF (debug only — must be `false` when EKF is active) |
| `max_marker_dist` | 2.5 m | Reject detections beyond this camera-to-marker distance |
| `min_perimeter_px` | 50.0 px | Reject detections with a pixel perimeter below this value |
| `cov_base_xy` | 0.01 m | Position uncertainty floor |
| `cov_dist_xy` | 0.02 m/m | Position uncertainty growth rate with distance |
| `cov_base_yaw` | 0.02 rad | Yaw uncertainty floor |
| `cov_dist_yaw` | 0.03 rad/m | Yaw uncertainty growth rate with distance |

---

## 8. Arena Marker Layout

The four markers are mounted on the arena walls at a height of 0.19 m above the floor, tilted 90° so that the flat marker surface faces downward into the arena. This allows the forward-facing robot camera to see the marker face at close-to-normal incidence.

| ID | x (m) | y (m) | Yaw | Arena position |
|---|---|---|---|---|
| 15 | 0.134 | 0.134 | +135° | South-west corner |
| 16 | 0.134 | 3.735 | +45° | North-west corner |
| 17 | 3.724 | 3.735 | −45° | North-east corner |
| 21 | 3.715 | 0.134 | −135° | South-east corner |

With this placement, the robot can typically see one or two markers at a time depending on its position and heading in the 3.9 × 3.9 m arena. The multi-marker fusion in §2.4 handles all cases where more than one is simultaneously visible, producing a single pose estimate with reduced uncertainty.

---

## 9. Integration with the Mission Orchestrator

**Stage 08 launch.** The orchestrator launches `aruco_localizer` via `ros2 launch aruco_localizer aruco_localizer.launch.py` and polls for a publisher on `/aruco_pose` with a configurable timeout (`amr_localizer.ready_timeout_sec`, default 30 s). The launch file also starts the RealSense driver and the static TF publishers for marker poses and camera extrinsics.

**VSLAM camera sharing.** When `vslam.enabled: true`, the orchestrator remaps the localizer's image and camera_info topics to the RealSense IR stream used by the cuSLAM pipeline (`vslam_image_topic`, `vslam_camera_info_topic`, `vslam_camera_frame`). This allows both the ArUco localizer and the cuSLAM pipeline to share the same physical camera without running two RealSense driver instances.

**EKF consistency.** The `world → odom` TF published by `alignment_node` in Stage 09.b ensures that the EKF's output `/amr/ekf/odom` is expressed in the same `world` frame as the ArUco pose. The two are therefore directly comparable; the EKF's correction step does not need to perform any frame change when incorporating `/aruco_pose`.

---

## 10. Known Limitations

**Occlusion.** If all four arena corners are occluded or outside `max_marker_dist`, no pose is published for that frame. Gaps in localization are handled by the EKF's prediction step (dead-reckoning from odometry and IMU), but drift accumulates during extended periods without ArUco updates.

**Height and tilt discarding.** The PnP solve produces a full 6-DoF estimate, but only $x$, $y$, and $\theta$ are forwarded to the EKF via the covariance mask. The z-axis and tilt estimates are discarded; this is appropriate for a ground robot operating on a flat surface but would require revision for non-planar terrains.

**`publish_tf` conflict with EKF.** The optional direct TF broadcast (`publish_tf: true`) emits a separate `world → aruco_base_footprint_est` transform that bypasses the EKF. Enabling it in production creates a second, unfiltered localization path that can produce confusing behavior in RViz and in downstream nodes that look up the TF tree. It is intended exclusively for offline debugging.

**Single dictionary and uniform marker size.** All markers must use the same ArUco dictionary and the same physical side length. Mixing dictionaries or sizes in one arena requires code modification; there is no per-ID size or dictionary override.