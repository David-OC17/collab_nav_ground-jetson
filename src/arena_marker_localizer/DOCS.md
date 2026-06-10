# Arena Marker Localizer
## Technical Documentation

---

## 1. Overview

`arena_marker_localizer` is a **ROS 2 service** (`/localize_markers`) that processes a
drone aerial video together with a synchronised OptiTrack pose log to produce
**world-frame poses of ArUco fiducial markers** on the arena floor.  It is the
authoritative source for marker **orientation** (yaw) consumed by the mission
orchestrator.

The package operates as a **batch service** — the complete scan video is processed
after the drone has landed.  A single service call blocks until all frames have been
processed; there is no incremental feedback.  The inputs are:

- A drone scan video (the same video that feeds `arena_map_builder`).
- The matching OptiTrack CSV (one row per video frame, time-aligned by row index).

The output per detected marker is a `MarkerPose` message containing:
- 3D pose in the map frame with 6×6 covariance (`PoseWithCovarianceStamped`)
- Convenience 2D pose (`Pose2D`) with `(x, y, θ)`
- Grid-cell indices and observation count

**Role in the mission:** The orchestrator fuses two independent sources for the goal
and AMR fiducial markers:

| Attribute | Source | Rationale |
|---|---|---|
| $(x, y)$ position | `arena_map_builder` stitched map | Positional bias in the PnP chain resists correction (see §8); stitcher metric projection is empirically more accurate for absolute position |
| Yaw $\theta$ | `arena_marker_localizer` PnP | Orientation is a purely geometric ratio of corner projections, not affected by map-frame origin shift |

---

## 2. Hardware: The 45° Mirror Rig

The DJI Tello's primary camera is forward-facing.  Its optical axis is parallel to the
drone's $+X_{\text{body}}$ axis; in the OpenCV camera frame ($+X_c$ right, $+Y_c$ down,
$+Z_c$ into scene) the forward camera looks along $+Z_c$.

To obtain a downward-facing view of the arena floor, a **45° mirror rig** is physically
attached below the Tello's nose.  The mirror redirects the optical axis from horizontal
to vertical-downward, effectively making the camera look straight down from the drone's
altitude.  The result is a nadir-like view of the arena despite the camera sensor
remaining in its original horizontal mounting.

The static transform `T_drone_from_cam` encodes the effective optical-centre position of
the redirected view relative to the OptiTrack rigid-body origin of the drone.  The
deployed calibrated values are:

```yaml
T_drone_from_cam:
  x    : -0.065   # m  (forward/backward offset from rigid-body origin)
  y    : -0.062   # m  (lateral offset)
  z    :  0.100   # m  (vertical offset — mirror sits ~10 cm below CoM)
  roll :  3.14159 # rad (π — flips camera +Z to point along drone −Z (downward))
  pitch:  0.0
  yaw  :  0.045556 # rad (2.6° heading misalignment from calibrate_bias_v3)
```

The `roll = π` rotation maps $+Z_c \to -Z_c$ and $+Y_c \to -Y_c$, converting the
standard OpenCV forward-facing camera convention into a downward-facing one aligned with
the drone body's $-Z_{\text{body}}$ axis.  The small residual yaw of 2.6° corrects a
mounting-angle misalignment between the mirror rig and the drone heading, fitted by
the bias calibration (§8).

---

## 3. Pipeline Overview

```
ALGORITHM: LocalizeMarkers(video_path, optitrack_csv)

Load intrinsics K, dist_coeffs
Load OptiTrack CSV → [DronePose₀ ... DronePoseₙ]
Precompute drone speed ṡᵢ (central finite difference) for velocity gate

obs ← {}   // per-marker observation lists

for i = 0..N-1 (stride s, max_workers threads):
    frame  ← video[i]
    pose_i ← csv[i]           // 1:1 by row index

    // Stage 1: Frame quality gate  (§4)
    if Blur(frame) < τ_blur or Artifact(frame) > τ_art: skip
    if ṡᵢ > v_max: skip           // velocity gate

    // Stage 2: Detection + PnP  (§5)
    detections ← ArUcoDetect(frame, K, dist)
    for det in detections:
        if det.reproj_err > τ_reproj: skip
        T_cam_marker ← det.T_cam_marker   // 4×4 from solvePnP

        // Stage 3: Transform chain  (§6)
        T_opti_drone ← BuildTransform(pose_i)
        T_map_marker ← T_map_from_opti ⊗ T_opti_drone ⊗ T_drone_from_cam ⊗ T_cam_marker
        pos ← T_map_marker[:3, 3]
        yaw ← ZYX_yaw(T_map_marker[:3,:3])

        obs[det.id].append(pos, yaw, drone_attitude)

// Stage 4 + 5: Aggregation  (§7)
results ← {}
for id, observations in obs:
    survivors ← MADGate(observations)       // §7.1
    pos_est   ← GeometricMedian(survivors)  // §7.2
    yaw_est   ← CircularMedian(survivors)   // §7.3
    results[id] ← MarkerResult(pos_est, yaw_est, covariance)

return results
```

---

## 4. Stage 1 — Frame Quality Gating

Frame quality gating runs identical logic to the `arena_map_builder` Stage 1 gate
(§3 of that document), minus the movement gate (irrelevant here since the goal is
marker visibility, not inter-frame alignment).

**Laplacian sharpness:**
$$\sigma^2_{\text{Lap}} = \mathrm{Var}(\nabla^2 I) \geq \tau_{\text{blur}} = 60$$

**DCT block-artifact ratio** — ratio of mean Sobel gradient energy on 8-pixel codec
block boundaries to off-boundary gradient energy:
$$r_{\text{art}} = \frac{\bar{g}_{\text{on-grid}}}{\bar{g}_{\text{off-grid}}} \leq \tau_{\text{art}} = 2.0$$

**Velocity gate** — frames where the OptiTrack-measured drone speed exceeds
$v_{\max} = 0.15$ m/s are dropped.  At that speed a multirotor tilts approximately 3–4°;
at 2 m altitude this projects to $\sim$10 cm of systematic marker position error per frame.
The gate primarily fires during U-turn transitions at the ends of lawnmower rows.

---

## 5. Stage 2 — ArUco Detection and PnP Pose Estimation

### 5.1 ArUco Corner Detection

ArUco markers (Garrido-Jurado et al., 2014) are planar fiducials consisting of a binary
pattern inside a black border.  Detection proceeds in four steps:

1. **Adaptive thresholding** — converts the grayscale frame to binary with locally
   adaptive thresholds, making detection robust to uneven illumination across the
   4 × 4 m arena.

2. **Contour extraction and quadrangle filtering** — connected components in the binary
   image are filtered by contour area, convexity, and minimum angular spacing of the four
   corners to find candidate squares.

3. **Subpixel corner refinement** — corner positions are refined to sub-pixel accuracy via
   intensity-gradient minimisation in a small neighbourhood.

4. **Bit extraction and dictionary lookup** — the region inside each candidate square is
   perspective-corrected, sampled on the bit grid, and compared against the configured
   dictionary (`DICT_4X4_50`, 50-marker 4×4-bit encoding).  A marker ID is assigned when
   the Hamming distance to the nearest codeword is within the error-correction capacity.

The deployed configuration uses `DICT_4X4_50` with a physical marker side length of
$s = 0.135$ m.

### 5.2 Object-Point Convention

The four corners of the marker are defined in the **marker frame** (origin at marker
centre, $+Z$ out of the front face):

$$\mathbf{X}_1 = \begin{pmatrix} -s/2 \\ +s/2 \\ 0 \end{pmatrix}, \quad
\mathbf{X}_2 = \begin{pmatrix} +s/2 \\ +s/2 \\ 0 \end{pmatrix}, \quad
\mathbf{X}_3 = \begin{pmatrix} +s/2 \\ -s/2 \\ 0 \end{pmatrix}, \quad
\mathbf{X}_4 = \begin{pmatrix} -s/2 \\ -s/2 \\ 0 \end{pmatrix}$$

These four 3D–2D correspondences $\{(\mathbf{X}_i, \mathbf{u}_i)\}$ are passed to the PnP
solver.

### 5.3 PnP Solver: IPPE_SQUARE

The Perspective-n-Point problem estimates the camera-from-marker transform $(R, \mathbf{t})$
such that:

$$\lambda_i \begin{pmatrix} \mathbf{u}_i \\ 1 \end{pmatrix} = K \begin{bmatrix} R \mid \mathbf{t} \end{bmatrix} \begin{pmatrix} \mathbf{X}_i \\ 1 \end{pmatrix}$$

where $K$ is the $3 \times 3$ camera intrinsic matrix and $\lambda_i > 0$ is the unknown
depth.

The solver used is **IPPE_SQUARE** (Infinitesimal Plane-Based Pose Estimation for square
targets, Collins & Bartoli 2014).  For a planar target all object points have $Z_i = 0$,
which reduces the 3D–2D correspondence problem to a **homography** between the marker
plane and the image plane:

$$\begin{pmatrix} u_i \\ v_i \\ 1 \end{pmatrix} \sim K \begin{bmatrix} \mathbf{r}_1 \mid \mathbf{r}_2 \mid \mathbf{t} \end{bmatrix} \begin{pmatrix} X_i \\ Y_i \\ 1 \end{pmatrix} = H \begin{pmatrix} X_i \\ Y_i \\ 1 \end{pmatrix}$$

$H$ can be estimated linearly from 4 correspondences using the DLT algorithm.  IPPE
then recovers $(R, \mathbf{t})$ from $H$ via a closed-form algebraic decomposition:
since $H = K[\mathbf{r}_1 \mid \mathbf{r}_2 \mid \mathbf{t}]$ and $\mathbf{r}_1, \mathbf{r}_2$
must be orthonormal columns of $R$, the decomposition reduces to a $2 \times 2$ SVD problem
giving **two candidate pose solutions**.

The `_SQUARE` variant exploits the 4-fold rotational symmetry of the square marker: the
second IPPE solution places the marker rotated by 180° about its face normal and is
geometrically distinct from the primary solution in all non-degenerate views.  The
ambiguity is resolved by selecting the solution with the lower reprojection error on the
full set of four corners.

**Advantages over iterative solvers (EPnP, LM):** IPPE_SQUARE is a closed-form
algebraic method — there are no convergence iterations, no local minima, and no
initialization sensitivity.  It is exact for noise-free inputs and numerically robust for
small perspective distortion, which is the dominant regime for nearly-nadir drone-floor
geometry.

### 5.4 Reprojection Error Gate

After the pose estimate $(R, \mathbf{t})$, the mean corner reprojection error is computed:

$$\varepsilon_{\text{reproj}} = \frac{1}{4} \sum_{i=1}^{4}
\left\| \pi(K, \text{dist}, R, \mathbf{t}, \mathbf{X}_i) - \mathbf{u}_i \right\|_2$$

where $\pi$ is the full lens-distortion projection.  Detections with
$\varepsilon_{\text{reproj}} > \tau_{\text{reproj}} = 4.0$ px are discarded before the
transform chain is evaluated.  This gate rejects detections where the corner positions
were corrupted by motion blur, partial occlusion, or decoding errors.

The output of this stage is `T_cam_from_marker` — the $4 \times 4$ homogeneous
transform placing the marker in the camera frame:

```
T_cam_from_marker = [ R  t ]
                    [ 0  1 ]
```

where $R = \text{Rodrigues}(\mathbf{r})$ converts the rotation vector returned by
`cv2.solvePnP` to a rotation matrix.

---

## 6. Stage 3 — Static Transform Chain

### 6.1 Chain Composition

Each per-frame observation of a marker produces one map-frame pose by composing four
transforms:

$$\mathbf{T}_{\text{map} \leftarrow \text{marker}} =
  \underbrace{\mathbf{T}_{\text{map} \leftarrow \text{opti}}}_{\text{static}}
  \cdot
  \underbrace{\mathbf{T}_{\text{opti} \leftarrow \text{drone}}(t)}_{\text{dynamic}}
  \cdot
  \underbrace{\mathbf{T}_{\text{drone} \leftarrow \text{cam}}}_{\text{static}}
  \cdot
  \underbrace{\mathbf{T}_{\text{cam} \leftarrow \text{marker}}(t)}_{\text{dynamic}}$$

Each transform is a $4 \times 4$ homogeneous matrix
$\mathbf{T} = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$.
The inversion identity $\mathbf{T}^{-1} = \begin{bmatrix} R^\top & -R^\top \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$
holds for all rigid transforms.

**Pieces of the chain:**

| Transform | Type | Source |
|---|---|---|
| $\mathbf{T}_{\text{cam} \leftarrow \text{marker}}(t)$ | Dynamic | `solvePnP` output, per frame |
| $\mathbf{T}_{\text{drone} \leftarrow \text{cam}}$ | Static | Config (`T_drone_from_cam`), calibrated once |
| $\mathbf{T}_{\text{opti} \leftarrow \text{drone}}(t)$ | Dynamic | OptiTrack CSV row $t$, built from position + rotation |
| $\mathbf{T}_{\text{map} \leftarrow \text{opti}}$ | Static | Config (`T_map_from_opti`), calibrated once |

### 6.2 OptiTrack Pose Formats and Rotation Representation

The OptiTrack CSV is read one row per video frame (strict 1:1 alignment by row index, not
by timestamp).  Three formats are supported, detected automatically from column headers:

| Format | Columns | Rotation representation |
|---|---|---|
| **Quaternion** (preferred) | `qx, qy, qz, qw` | $\mathbf{R} = \text{quat2R}(q_x, q_y, q_z, q_w)$ — no Euler singularities |
| **Full Euler** (legacy) | `roll, pitch, yaw` | $\mathbf{R} = R_z(\psi) R_y(\phi) R_x(\rho)$ — ZYX Tait-Bryan |
| **Yaw-only** (oldest) | `yaw` | $\mathbf{R} = R_z(\psi)$ — pure heading, roll=pitch=0 |

The quaternion format is strongly preferred because it avoids the gimbal-lock singularity
at $|\text{pitch}| = 90°$ and eliminates floating-point drift in Euler angle conventions.
During recording the `record_scan` utility logs quaternions directly from OptiTrack's VRPN
stream.

The quaternion-to-rotation-matrix conversion:
$$R = \begin{pmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{pmatrix}$$

Quaternions are normalised to unit length on load to guard against logging drift:
$\hat{q} = q / \|q\|_2$.

### 6.3 Coordinate Convention and Axis Alignment

All rotations use **intrinsic ZYX Tait-Bryan** (yaw→pitch→roll):
$$\mathbf{R}(\rho, \phi, \psi) = R_z(\psi)\, R_y(\phi)\, R_x(\rho)$$

The `T_map_from_opti` transform encodes the mapping from OptiTrack world coordinates to
the nav2 map frame (bottom-left origin, $+X$ right, $+Y$ up).  The deployed configuration
has `yaw = π/2` (90°), which rotates OptiTrack $+X$ onto map $+Y$ and OptiTrack $+Y$
onto map $-X$ — reflecting the physical alignment of the arena in the motion-capture
volume.  The translation $(x=1.881, y=1.948)$ m places the arena corner in the
OptiTrack frame.

The per-axis sign flips (`x_dir`, `y_dir`) are applied as a right-multiplied diagonal
matrix to `T_map_from_opti` and allow correcting for OptiTrack axis conventions that
differ from the arena map without modifying the calibrated rotation.

---

## 7. Stage 4 — Observation Accumulation

For each video frame that passes the quality and velocity gates and yields at least one
valid PnP detection, the map-frame position and yaw are appended to a per-marker
observation list.

Each marker also stores the drone's **map-frame attitude** $(r_d, \phi_d, \psi_d)$ at
the time of each observation.  These are stored — not averaged immediately — so that
`calibrate_bias_v3` can reconstruct the full rotation matrix $\mathbf{R}_{\text{map}
\leftarrow \text{drone}}(t)$ for each observation and build the attitude-diversity design
matrix (§8.2).

The observation list is capped at `max_obs_per_marker = 200` via a FIFO ring: if more
observations arrive, the oldest are discarded.  This bounds memory regardless of scan
length.  In practice a typical lawnmower scan yields **120–180 observations** per
marker, well above the aggregation minimum of 50.

---

## 8. Stage 5 — Outlier Rejection and Aggregation

### 8.1 MAD Gate (Per-Axis + Circular Yaw)

Before the geometric median is computed, outlier observations are removed using a
per-axis Median Absolute Deviation gate.  An observation is rejected if it is an outlier
in **any single axis** — position or yaw — so that position and yaw are always estimated
from the same survivor set.

For each Cartesian axis $j \in \{x, y, z\}$:

$$m_j = \mathrm{median}_i(p_{ij}), \qquad \text{MAD}_j = \mathrm{median}_i(|p_{ij} - m_j|)$$

$$\text{reject observation } i \iff |p_{ij} - m_j| > k \cdot \text{MAD}_j, \quad k = 3.5$$

For yaw (a circular quantity), the gate uses a wrapped residual relative to the circular
median $\hat{\theta}$:

$$\text{res}_i = \mathrm{wrap}(\theta_i - \hat{\theta}), \qquad
\text{MAD}_\theta = \mathrm{median}_i(|\text{res}_i|)$$

$$\text{reject observation } i \iff |\text{res}_i| > k \cdot \text{MAD}_\theta$$

where $\mathrm{wrap}(\alpha) = (\alpha + \pi) \bmod 2\pi - \pi$ maps any angle to $[-\pi, \pi]$.

A marker is rejected entirely if fewer than `min_observations = 50` survive the gate.

### 8.2 Geometric Median via Weiszfeld Iteration

The Cartesian position is estimated using the **geometric median** of the surviving
3D observations.  The geometric median minimises the sum of Euclidean distances
(as opposed to the mean, which minimises the sum of squared distances):

$$\hat{\mathbf{p}} = \arg\min_{\mathbf{x}} \sum_{i=1}^{M} \|\mathbf{x} - \mathbf{p}_i\|_2$$

The geometric median has a **breakdown point of ~50%** — up to half the observations
can be arbitrary outliers without corrupting the estimate — and is more robust than the
coordinate-wise median when noise is anisotropic (which it is here, since the camera's
depth axis contributes more error than the lateral axes).

The **Weiszfeld algorithm** iterates:

$$\mathbf{x}^{(k+1)} = \frac{\displaystyle\sum_{i:\, \mathbf{p}_i \neq \mathbf{x}^{(k)}} w_i^{(k)}\, \mathbf{p}_i}{\displaystyle\sum_{i:\, \mathbf{p}_i \neq \mathbf{x}^{(k)}} w_i^{(k)}}, \qquad w_i^{(k)} = \frac{1}{\|\mathbf{p}_i - \mathbf{x}^{(k)}\|_2}$$

Initialised at the centroid $\mathbf{x}^{(0)} = \bar{\mathbf{p}}$; converges when
$\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|_2 < \varepsilon = 10^{-5}$ m or after
100 iterations.

```
ALGORITHM: Weiszfeld(points P, ε, max_iter)
x ← mean(P)
for k = 1..max_iter:
    dists ← ‖P − x‖₂  (per row)
    nonzero ← { i : dists[i] > 1e-9 }
    w ← 1 / dists[nonzero]
    x_new ← (P[nonzero]ᵀ w) / sum(w)    // weighted mean
    if ‖x_new − x‖ < ε: return x_new
    x ← x_new
return x
```

### 8.3 Yaw Aggregation (Circular Median and Dispersion)

Yaw is a circular quantity; arithmetic averaging produces biased results near the $\pm\pi$
wrap boundary.  The circular median is estimated via the **median of the unit-vector
projection**:

$$\hat{\theta} = \arctan2\!\left(\mathrm{median}_i(\sin\theta_i),\; \mathrm{median}_i(\cos\theta_i)\right)$$

The circular dispersion (variance proxy) is the **concentration measure** $\bar{R}$:

$$\bar{R} = \sqrt{\overline{\sin\theta}^2 + \overline{\cos\theta}^2}, \qquad
  \hat{\sigma}^2_\theta = -2\ln\bar{R}$$

$\bar{R} = 1$ means all observations are identical ($\hat{\sigma}^2_\theta = 0$);
$\bar{R} \to 0$ means observations are uniformly distributed ($\hat{\sigma}^2_\theta \to \infty$).
This is the standard circular dispersion estimator (Mardia & Jupp 2000).

### 8.4 Output Covariance

The service publishes per-marker covariance for downstream EKF/fusion consumers.
Position covariance is the **sample covariance of surviving observations** scaled by
observation count:

$$\hat{\Sigma}_{xx} = \frac{\text{Var}(p_x^{\text{inliers}})}{n}, \quad
  \hat{\Sigma}_{yy} = \frac{\text{Var}(p_y^{\text{inliers}})}{n}, \quad
  \hat{\Sigma}_{xy} = \frac{\text{Cov}(p_x, p_y)_{\text{inliers}}}{n}$$

Yaw covariance uses the circular dispersion scaled by $n$:
$\hat{\sigma}^2_\theta / n$.

The full $6 \times 6$ covariance matrix in `PoseWithCovarianceStamped` has large values
($10^6$) for $z$, roll, and pitch — those degrees of freedom are not estimated (the
marker is assumed flat on the floor, roll=pitch=0).

---

## 9. Error Sources and Bias Calibration

### 9.1 Error Decomposition

The map-frame position error of a single PnP observation decomposes into four additive
contributions:

$$\boldsymbol{\varepsilon}_{\text{map}} =
  \underbrace{\begin{pmatrix} c_x \\ c_y \end{pmatrix}}_{\text{(A) map origin shift}}
  + \underbrace{\mathbf{R}_{\text{map} \leftarrow \text{drone}}^{[2 \times 3]} \begin{pmatrix} dx_d \\ dy_d \\ dz_d \end{pmatrix}}_{\text{(B) camera lever arm}}
  + \underbrace{s\, \hat{\mathbf{p}}_{\text{est}}}_{\text{(C) radial scale bias}}
  + \underbrace{\delta\psi \cdot \mathbf{R}_\perp \hat{\mathbf{p}}_{\text{est}}}_{\text{(D) yaw misalignment}}$$

**(A) Map origin shift $(c_x, c_y)$:** The `T_map_from_opti` translation places the
map coordinate origin in the OptiTrack world.  Any error in the measured arena-corner
position propagates as a constant bias into all marker positions.

**(B) Camera lever arm $(dx_d, dy_d, dz_d)$:** The camera's effective optical centre
(at the mirror) is not coincident with the OptiTrack rigid-body origin.  This offset,
expressed in the drone frame, projects into map-frame error through the instantaneous
rotation $\mathbf{R}_{\text{map} \leftarrow \text{drone}}(t)$.  Because the rotation
matrix changes with drone attitude, this term is **not constant across observations** —
it is the only term that is identifiable from attitude diversity.

**(C) Radial scale bias $s$:** A systematic scale error proportional to estimated
distance from the map origin.  Physical causes: wrong `marker_size_m` configuration, or
systematic error in the flying altitude (OptiTrack altitude miscalibration).

**(D) Yaw misalignment $\delta\psi$:** A residual heading offset between the OptiTrack
frame and the map frame, causing a rotation of all estimated positions around the map
origin.  Absorbed by the `T_map_from_opti.yaw` parameter.

### 9.2 Design Matrix and Linear Least Squares

For $N$ (marker, scan-video) observation pairs, the model is cast as a linear system
$\mathbf{A}\boldsymbol{\theta} = \mathbf{b}$, where $\mathbf{b} = \mathbf{p}_{\text{gt}} - \mathbf{p}_{\text{est}}$
is the measured correction vector and $\boldsymbol{\theta} = [c_x, c_y, dx_d, dy_d, dz_d]^\top$
(optionally with scale $s$ appended).

The design matrix $\mathbf{A} \in \mathbb{R}^{2N \times 5}$ has two rows per observation
(x and y residual equations):

$$\mathbf{A}_{2i,\,:} = \begin{pmatrix} 1 & 0 & R_{11} & R_{12} & R_{13} \end{pmatrix}$$
$$\mathbf{A}_{2i+1,\,:} = \begin{pmatrix} 0 & 1 & R_{21} & R_{22} & R_{23} \end{pmatrix}$$

where $R_{jk}$ are the top-two rows of $\mathbf{R}_{\text{map} \leftarrow \text{drone}}(t_i)$
evaluated at the mean drone attitude over the observation's inlier set.

### 9.3 Physically Bounded Fitting with IRLS

Parameter bounds derived from physical constraints:

| Parameter | Bound | Physical justification |
|---|---|---|
| $c_x, c_y$ | $\pm 5.0$ m | Map origin cannot be more than one arena width away |
| $dx_d, dy_d, dz_d$ | $\pm 0.10$ m | Tello camera is physically $< 10$ cm from CoM |
| $s$ (optional) | $\pm 0.30$ | Scale correction beyond 30% indicates a misconfiguration |

Fitting uses **Iteratively Reweighted Least Squares (IRLS)** with a Huber-like weight
function to downweight gross outlier observations (e.g. scans where a marker was
partially occluded):

```
ALGORITHM: IRLS_Bias_Fit(A, b, lo, hi, σ, max_iter)
w ← ones(N)          // per-observation weights
for k = 1..max_iter:
    W ← diag(repeat(w, 2))     // broadcast: 2 rows per obs
    θ ← BoundedLeastSquares(W·A, W·b, lo, hi)   // bvls
    r ← b − A θ                                 // residuals (2N,)
    pos_err_i ← sqrt(r[2i]² + r[2i+1]²)        // per-obs L2 error (N,)
    w_new ← σ / max(pos_err_i, σ)               // σ = 0.30 m threshold
    if ‖w_new − w‖∞ < 1e-4: break
    w ← w_new
return θ
```

The weight $w_i = \sigma / \max(r_i, \sigma)$ is $1$ for well-fitting observations and
shrinks proportionally for observations with residual $> \sigma = 0.30$ m.

### 9.4 Attitude Degeneracy Analysis

The terms $(dx_d, dy_d, dz_d)$ are identifiable only when the drone attitude varies
sufficiently across observations.  If all scans are hover-only (constant pitch and roll),
the rotation matrices $\mathbf{R}^{[2 \times 3]}$ are nearly identical for all
observations — the columns 2–4 of $\mathbf{A}$ (the attitude sub-block) become nearly
rank-deficient and the solver absorbs the camera offset into $(c_x, c_y)$, fitting
training data well but generalising poorly.

**v2 bug:** Loose bounds of $\pm 30$ cm on $(dx_d, dy_d)$ allowed the solver to exploit
this near-degeneracy — the fitted $(dx_d, dy_d)$ were physically implausible but made the
system $\mathbf{A}\boldsymbol{\theta} \approx \mathbf{b}$ on training data.  Mean
leave-one-out CV error: **54 cm**.

**v3 fix:** (1) Bound $(dx_d, dy_d, dz_d)$ to $\pm 10$ cm.  (2) Check the SVD condition
number of the attitude sub-block $\mathbf{A}[:, 2:5]$; if $\kappa > 50$, fall back to a
2-parameter model $(c_x, c_y)$ only and print an actionable diagnostic.  (3) Compute hat-matrix
leverage scores $h_i = [\mathbf{Q}\mathbf{Q}^\top]_{ii}$ (where $\mathbf{A} = \mathbf{Q}\mathbf{R}$)
to identify which scans contribute most information.  (4) Require $\geq 6$
(marker × scan) pairs and $\geq 30°$ spread in drone pitch for $(dz_d)$
observability.

```
ALGORITHM: AttitudeDegeneracyCheck(A_full)
_, sv, _ ← SVD(A_full[:, 2:5])
κ ← sv[0] / max(sv[-1], 1e-12)
if κ > 50:
    fit only [c_x, c_y]
    warn "add dynamic scans with varied pitch/roll"
else:
    fit full [c_x, c_y, dx_d, dy_d, dz_d]
```

### 9.5 Cross-Validation and Calibration Lifecycle

Calibration is run **once per arena configuration** using a dedicated set of manually
recorded scans (not mission scan videos).  The manual scans are chosen to provide
attitude diversity: the pilot deliberately varies drone pitch and roll by
$\geq 30°$ across multiple scans, providing the leverage that hover-only lawnmower
scans cannot.

`calibrate_bias_v3` runs k-fold cross-validation (leave-one-scan-out by default) to
report the **generalisation position error** — the error on held-out scans not used
for fitting.  The fitted correction is absorbed back into `T_map_from_opti` and
`T_drone_from_cam` and written to `default.yaml`.  The `_kfold.yaml` companion file
records fold-level diagnostics for review.

> **Known limitation:** Even after v3 calibration, the absolute position accuracy of the
> PnP-derived marker positions is insufficient for reliable path-planning goal
> localisation.  The residual error is dominated by sources outside the linear model:
> (1) **mirror optical distortion** — the 45° mirror introduces its own aberrations
> that are not captured by the standard pinhole + radial/tangential distortion model
> calibrated on the bare lens; (2) **pitch and roll effects through the mirror** — small
> drone tilt changes the effective viewing geometry through the mirror in a way that is
> not fully accounted for by the rigid `T_drone_from_cam` model; (3) **varying drone
> altitude** — the lawnmower trajectory has measurable altitude variation across the
> arena, producing a depth-dependent scale error that is only partially captured by the
> optional scale term $s$; (4) **OptiTrack transient failures** — the motion-capture
> system occasionally produces physically implausible position and velocity jumps (wrong
> rigid-body assignments or occluded markers) for 1–2 seconds at a time; these frames
> pass the velocity gate only if the outlier is brief enough and inject large position
> errors that survive the MAD gate when they cluster.
> Consequently, the mission orchestrator uses **stitcher-derived positions** and only
> takes **yaw from this service** (§10).

---

## 10. Orientation: Why Yaw but Not Position

The mission orchestrator consumes:

- **$(x, y)$ position** from `arena_map_builder`'s stitched map (color centroid detection on
  the composited image, metric-scaled via background.png).
- **Yaw $\theta$** from this service's PnP chain.

The rationale is geometrical:

**Position error** in the PnP chain accumulates from four sources (§9.1), the dominant
of which is the map-frame origin shift $(c_x, c_y)$.  This shift is not directly
observable from the marker images — it must be inferred from known ground-truth positions
collected offline.  Even a 3 cm calibration residual in $(c_x, c_y)$ propagates equally
to all markers and cannot be corrected at runtime.

**Yaw error** from the PnP chain has a different character.  The yaw of the marker pose
is determined by the ratio of corner projections — it is a **relative geometry** measure
within the image.  Map-frame origin shift $(c_x, c_y)$ translates all positions uniformly
but does **not** rotate the estimated marker orientation.  The dominant yaw error is
the residual $\delta\psi$ in `T_map_from_opti.yaw`, which `calibrate_bias_v3` fits
accurately as a circular mean of angular residuals.  After calibration, the yaw from
this service is reliable for orienting the AMR approach.

The flat-floor assumption (roll=pitch=0 in the output pose) holds because all arena
markers are placed on the floor and the map is a 2D occupancy grid — the $z$ coordinate
and tilt of the marker are not consumed.

---

## 11. Service Interface

`arena_marker_localizer/srv/LocalizeMarkers`:

```
# Request
string video_path
string optitrack_csv
---
# Response
bool                                  success
string                                message
arena_marker_localizer/MarkerPose[]   markers
```

`arena_marker_localizer/msg/MarkerPose`:

```
uint32                           id
geometry_msgs/Pose               pose_3d        # roll=pitch=0; yaw only
geometry_msgs/Pose2D             pose_2d        # x, y, theta convenience
sensor_msgs/PoseWithCovariance   pose_with_covariance  # 6×6 matrix
uint32                           cell_x
uint32                           cell_y
uint32                           n_observations
```

Key parameters (all configurable as ROS 2 node parameters, loaded from `default.yaml`):

| Parameter | Deployed value | Effect |
|---|---|---|
| `dictionaries` | `["DICT_4X4_50:0.135"]` | ArUco family + physical marker size |
| `max_reproj_err_px` | 4.0 px | Per-detection PnP quality gate |
| `aggregation.mad_k` | 3.5 | MAD gate factor (~3.5σ equivalent) |
| `aggregation.min_observations` | 50 | Minimum inlier count to accept a marker |
| `max_obs_per_marker` | 200 | FIFO cap on observation list |
| `processing.max_workers` | 5 | Parallel frame processing threads (Jetson: 6 cores) |
| `processing.frame_stride` | 1 | Process every frame |
| `processing.max_drone_velocity_m_s` | 0.15 | Drop frames above this speed (U-turn gate) |
| `quality.blur_thresh` | 60.0 | Minimum Laplacian variance |
| `quality.artifact_thresh` | 2.0 | Maximum DCT block-artifact ratio |

---

## 12. Figure Placeholders

> **Figures to add for final report:**
> - Hardware diagram: 45° mirror rig on Tello with `T_drone_from_cam` dimensions annotated
> - Transform chain diagram: four transforms with frame axes drawn at each node
> - IPPE_SQUARE geometry: marker frame, image plane, two candidate solutions, selection criterion
> - Aggregation pipeline diagram: scatter of raw observations → MAD gate survivors → geometric median
> - Calibration scatter plot: GT vs. estimated positions before / after v3 correction, per scan
> - CV fold error bars: per-fold RMS position error showing generalisation performance
> - Attitude leverage scores: bar chart of leverage per scan, showing which scans add most information
