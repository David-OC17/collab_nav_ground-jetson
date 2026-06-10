# Arena Map Builder
## Technical Documentation

---

## 1. Overview

`arena_map_builder` transforms a monocular aerial video recorded by the DJI Tello into a
**metrically-scaled, probabilistic 2D occupancy grid** suitable for direct consumption by a
nav2-compatible path planner.  It is the core perception output of the aerial phase: the
drone surveys the arena from above and the stitched map becomes the AMR's world model.

The package operates as a **ROS 2 action server** (`BuildArenaMap`).  The mission
orchestrator fires the action at stage 03 (drone scan) and joins at stage 04 after the drone
has landed.  The action accepts a path to the recorded video, runs the full pipeline, and
returns:

- A `nav_msgs/OccupancyGrid` at 5 cm/cell resolution
- Estimated world-frame positions of the goal ArUco marker and the AMR fiducial marker
- A 46-element diagnostic feature vector used by the mission orchestrator's map-quality
  classifier (§10) to decide whether the map meets the quality threshold for navigation

**Pipeline summary** (five sequential stages):

```
(1) Frame Extraction → (2) Incremental Stitching → (3) Obstacle Transfer
                    → (4) Consistency Scoring   → (5) Occupancy Rasterization
```

---

## 2. Arena-Specific Design Constraints

The pipeline encodes several assumptions about the physical arena.  These were manually
calibrated for the specific test environment and represent hard limitations that must be
recalibrated for any different arena.

| Constraint | Value | Purpose |
|---|---|---|
| Blue adhesive tape grid on floor | HSV hue 90–130 | Grid intersection detection; feature-exclusion mask |
| Brown perimeter wall | HSV hue 5–25, S 50–180 | Arena boundary detection for occupancy origin |
| Pink/magenta obstacle color in stitched output | BGR (255, 0, 255) | Obstacle blob extraction |
| Green wall color in stitched output | BGR (0, 255, 0) | Convex-hull cropping |
| Solid-color ArUco recoloring (goal: red, AMR: cyan) | BGR (0,0,255) / (255,255,0) | Marker localization in final map |
| `background.png` clean template | Hand-generated top-down view | Obstacle projection reference frame |
| Arena dimensions | 4.0 × 4.0 m | Metric scale calibration in occupancy grid |

The **background template** (`background.png`) is a hand-generated top-down image of the
empty arena floor, showing the brown wall boundary and blue tape grid at the correct aspect
ratio.  It serves as the canonical reference frame onto which detected obstacles are
projected.  It is a static asset produced once per arena configuration.

These color-based assumptions make the pipeline brittle to lighting changes and arena
reconfiguration.  They represent a known limitation of the current implementation.

---

## 3. Stage 1 — Frame Extraction and Filtering

### 3.1 Acquisition Modes

The pipeline supports two intake modes that share identical filtering logic:

**Offline mode** operates on a saved video file after the drone has landed.  A streaming
generator yields one filtered frame at a time — no full frame list is ever held in RAM.
This is the traditional post-landing processing path.

**Online mode** (default in the deployed mission) receives frames from the live camera
stream during the flight via a push-model gate (`OnlineFrameGate`).  Each incoming frame
is evaluated independently; accepted frames are placed onto the growing canvas
immediately.  This hides the stitching latency behind the flight duration: the map is
largely complete by the time the drone lands, and the final consistency pass is the only
remaining computation.  The quality and movement gates are identical in both modes; only
the delivery mechanism differs (pull-from-file vs. push-from-stream).

### 3.2 Frame Sub-sampling

Frames are sub-sampled from the source video at a configurable target rate (default 5 fps).
In offline mode this is implemented as a 1-in-$k$ stride over decoded frames.  In online
mode a timestamp-based throttle rejects frames faster than the target interval.

### 3.3 Quality Gate

Each candidate frame passes three quality checks before entering the stitching stage:

1. **Brightness** — mean grayscale value must lie in $(l_{\min}, l_{\max})$.  Frames outside
   this range are under- or over-exposed and yield degenerate feature descriptors.

2. **Laplacian sharpness** — the variance of the Laplacian response (a focus measure)
   must exceed a threshold $\tau_{\text{blur}}$:
   $$\sigma^2_{\text{Lap}} = \mathrm{Var}\!\left(\nabla^2 I\right) \geq \tau_{\text{blur}}$$
   Blurry frames from drone motion blur or out-of-focus moments are rejected.

3. **DCT block-artifact ratio** — the ratio of mean pixel difference at 8-pixel-spaced
   block boundaries (both axes) to the overall mean neighbour difference detects H.264/H.265
   macro-block compression artifacts that corrupt feature descriptors near boundaries.

### 3.4 Movement Gate

The movement gate serves two purposes: it skips **static frames** (drone hovering without
translating — redundant information) and drops **jerk frames** (excessive displacement
indicating flight instability or tracking loss).

The primary estimator uses **median displacement** of nearest-neighbour feature matches,
normalized by the frame diagonal:

$$d_{\text{med}} = \frac{\mathrm{median}_m \left\| \mathbf{p}^{(1)}_m - \mathbf{p}^{(2)}_m \right\|}{D_{\text{frame}}}$$

The median is used — not the mean — because on repetitive grid scenes the ratio test
discards most matches, and the remaining nearest-neighbour matches contain many
wrong-cell correspondences.  The median displacement is naturally robust to these outliers
since the true motion cluster forms the majority.

A frame is kept iff $d_{\min} \leq d_{\text{med}} \leq d_{\max}$ with defaults
$d_{\min} = 0.015$, $d_{\max} = 0.55$.

When fewer than 8 feature matches exist (low-texture hover sections), a **thumbnail
pixel-difference fallback** computes the mean absolute difference between a 160×90
grayscale thumbnail of the current frame and the last accepted frame.  A difference below
a threshold $\tau_{\text{pix}}$ classifies the frame as static.

---

## 4. Stage 2 — Incremental Frame Stitching

The core loop processes accepted frames one at a time.  For each new frame it selects the
best reference candidate from a temporal pool, estimates the transformation mapping the new
frame into the canvas coordinate system, and blends it onto the pre-allocated canvas.  The
high-level algorithm is:

```
ALGORITHM: IncrementalStitch(frame_stream)

canvas   ← allocate(8000 × 8000)         // §4.5
coverage ← zeros(8000 × 8000, bool)
ring_buf ← RingBuffer(capacity = L)      // recent frames
keyframes ← []                           // long-range anchors

for frame in frame_stream:               // §3 quality-gated
    feats_cur ← ExtractFeatures(frame)   // §4.2

    best_H, best_n_in ← None, 0
    for ref in ring_buf ∪ keyframes:     // §4.4 — parallel search
        H, n_in ← PairwiseAlign(feats_ref[ref], feats_cur)   // §4.3
        if n_in > best_n_in AND ValidateGeometry(H, canvas):
            best_H, best_n_in ← H, n_in

    if best_H is None: skip frame

    H_canvas ← H_canvas_ref ∘ best_H    // compose into canvas frame

    WarpAndBlend(frame, H_canvas, canvas, coverage)   // §4.5–4.6

    ring_buf.push(frame)
    if frame_index % keyframe_interval == 0:
        keyframes.append(frame)          // cap at 20

return canvas, coverage
```

### 4.1 The Fundamental Alignment Choice: Similarity over Full Homography

Top-down drone footage at roughly constant altitude is geometrically a **2D rigid-body
motion problem**: translation, rotation, and a small uniform scale change from altitude
variation.  This has 4 degrees of freedom.

A full projective homography has 8 degrees of freedom.  Fitting it to a 4-DoF scene gives
RANSAC the freedom to explain wrong-cell matches on repetitive grid structure with small
perspective terms.  These spurious perspective components look locally valid (positive
perspective divide, convex warped quad) in any single frame, but **compound across
hundreds of frames** into fan/ray artifacts and global map distortion.

The pipeline therefore estimates a **similarity transform** (translation + rotation +
uniform scale, 4 DoF).  In homogeneous coordinates:

$$\mathbf{H}_{\text{sim}} = \begin{pmatrix} a & -b & t_x \\ b & \phantom{-}a & t_y \\ 0 & 0 & 1 \end{pmatrix}, \qquad a = s\cos\theta,\quad b = s\sin\theta$$

where $s \in \mathbb{R}^+$ is a uniform scale factor (altitude variation), $\theta$ is the
in-plane rotation (heading drift), and $(t_x, t_y)$ is translation.  The constraint
$\mathbf{H}[2,:] = [0,\,0,\,1]$ enforces that parallel lines remain parallel and that
there is no projective distortion — exactly the geometry seen from directly above.

`cv2.estimateAffinePartial2D` returns the $2 \times 3$ sub-matrix; the bottom row
$[0,\,0,\,1]$ is appended to produce a $3 \times 3$ matrix compatible with downstream
`warpPerspective`, homography composition, and canvas-expansion code.

### 4.2 Feature Extraction

#### 4.2.1 SIFT — Production Backend

**SIFT** (Scale-Invariant Feature Transform, Lowe 2004) is the production extractor,
configured with a budget of 5000 keypoints per frame.

**Keypoint detection** operates on a Difference-of-Gaussian (DoG) pyramid.  The image is
convolved with Gaussians at logarithmically-spaced scales $\sigma_k = \sigma_0 \cdot 2^{k/n}$,
and the DoG between adjacent scales approximates the scale-normalised Laplacian of Gaussian:

$$\text{DoG}(x, y, \sigma_k) \approx \sigma^2 \nabla^2 G * I$$

Local extrema of the DoG volume across scale and space are candidate keypoints.  Each is
localised to sub-pixel accuracy via quadratic interpolation of the 3D DoG neighbourhood,
then filtered by a contrast threshold (removes noise responses) and a principal-curvature
ratio test (removes edge responses; Hessian eigenvalue ratio $< 10$).

**Orientation assignment** examines the gradient magnitude and orientation histogram in a
$4.5\sigma$-radius neighbourhood and assigns the dominant peak, making the descriptor
invariant to image rotation.

**Descriptor computation** samples a $16 \times 16$ patch around each keypoint (at its
characteristic scale and orientation), divides it into a $4 \times 4$ grid of cells, and
accumulates an 8-bin gradient orientation histogram per cell.  Concatenating the
$4 \times 4 \times 8 = 128$ histogram values and L2-normalising yields the
**128-dimensional SIFT descriptor**.

#### 4.2.2 Feature-Exclusion Masking

The blue adhesive tape grid concentrates feature detectors on a periodic structure.
Keypoints on the tape have near-identical descriptors (all are samples of the same
repeating pattern at slightly different positions) and produce false matches to different
tape segments rather than the same physical point.

An HSV mask is computed, dilated by $d_{\text{px}}$ pixels, and inverted to produce a
binary eligibility mask passed to `SIFT.detectAndCompute`:

$$M_{\text{excl}}(p) = \begin{cases}
0 & \text{if } \mathrm{HSV}(p) \in
    [\mathrm{H}_{90}^{130},\,\mathrm{S}_{60}^{255},\,\mathrm{V}_{60}^{255}]
    \oplus \mathcal{B}(d_{\text{px}}) \\
1 & \text{otherwise}
\end{cases}$$

where $\oplus \mathcal{B}(d_{\text{px}})$ denotes morphological dilation by a disk of
radius $d_{\text{px}}$ (default 5 px).  Confirmed by the v2 sweep as non-negotiable:
without it RANSAC converges on wrong-cell hypotheses.

Only SIFT supports this mask.  SuperPoint has no mask input API.

#### 4.2.3 SuperPoint — GPU Alternative Backend

**SuperPoint** (DeTone et al., 2018) is a self-supervised convolutional network with a
shared VGG-style encoder and two independent decoder heads — one for interest-point
detection, one for dense descriptor computation.  It is evaluated in the v3 sweep
(Group C) as a GPU-accelerated alternative.

The model is deployed as a static-input ONNX graph (batch size 2, fixed at export time).
For each frame:

```
ALGORITHM: SuperPointExtract(frame, model_H, model_W)

gray ← BGR_to_grayscale(frame)
gray ← resize(gray, model_W × model_H)         // static input shape requirement
pp   ← gray / 255.0  reshaped to (1,1,mH,mW)
pair ← concat([pp, pp], axis=0)                // (2,1,mH,mW) — batch=2

kp_model, _, des ← onnx_session.run({"images": pair})
    // kp_model : (K, 2)     — keypoint (x,y) in model space
    // des      : (1, K, 256) — L2-normalised 256-dim descriptors

kp_px   ← kp_model * [W_orig/model_W, H_orig/model_H]   // scale to frame pixels
kp_norm ← 2·kp_px / [W_orig, H_orig] − 1               // normalise to [−1, 1]

return kp_px, kp_norm, des
```

SuperPoint descriptors are **256-dimensional** (vs SIFT's 128) and share the same Lowe
ratio test + mutual cross-check matching path (L2 distance on float32 descriptor vectors).
The v3 Group C sweep evaluates the trade-off between representational capacity and the
absence of blue-tape suppression.

### 4.3 Descriptor Matching Pipeline and Outlier Rejection

The full pairwise alignment procedure:

```
ALGORITHM: PairwiseAlign(feat_ref, feat_cur, match_ratio r, mad_factor κ)

// ── Step 1: Descriptor matching ──────────────────────────────────────────
matches_12 ← BFMatcher(L2).knnMatch(des_ref, des_cur, k=2)

// Lowe's ratio test (forward direction):
good_12 ← { m  |  m.distance < r · n.distance,  for (m,n) in matches_12 }

// Mutual cross-check (backward direction must also pass ratio test):
matches_21 ← BFMatcher(L2).knnMatch(des_cur, des_ref, k=2)
back ← { m.queryIdx → m.trainIdx  |  m.distance < r · n.distance,
                                      for (m,n) in matches_21 }
good ← { m ∈ good_12  |  back[m.trainIdx] == m.queryIdx }

if |good| < 12: return (None, 0)

pts_ref ← keypoints_ref[ m.queryIdx for m in good ]
pts_cur ← keypoints_cur[ m.trainIdx for m in good ]

// ── Step 2: MAD displacement pre-filter ─────────────────────────────────
Δᵢ   ← pts_ref[i] − pts_cur[i]          // per-match 2D displacement vector
μ    ← columnwise_median(Δ)             // robust central motion estimate
devᵢ ← ‖Δᵢ − μ‖₂                        // deviation from median motion
MAD  ← median(dev) + ε                  // ε = 1e-3 for numerical stability
keep ← { i  |  devᵢ < κ · MAD }        // κ = 4.0 (deployed baseline)

if |keep| < 12: return (None, 0)

// ── Step 3: Similarity RANSAC ────────────────────────────────────────────
M, inliers ← estimateAffinePartial2D(
    pts_cur[keep], pts_ref[keep],
    method      = RANSAC,
    threshold   = 3.0 px,
    maxIters    = 5000,
    confidence  = 0.999,
    refineIters = 50
)

if M is None or |inliers| < 8: return (None, 0)

H ← vstack([ M ; 0 0 1 ])    // promote 2×3 affine → 3×3

// ── Step 4: Geometry validation ─────────────────────────────────────────
// Reject if frame centre maps > 6 frame-widths from origin,
// or if composed H places frame outside canvas bounds.
if not ValidateGeometry(H, frame_w, frame_h): return (None, 0)

return (H, |inliers|)
```

#### 4.3.1 Lowe's Ratio Test

For each query descriptor $\mathbf{d}_q$, the two nearest descriptors in the target set
are retrieved: $\mathbf{d}_{m_1}$ (best) and $\mathbf{d}_{m_2}$ (second-best).  The match
is accepted iff:

$$\frac{\|\mathbf{d}_q - \mathbf{d}_{m_1}\|_2}{\|\mathbf{d}_q - \mathbf{d}_{m_2}\|_2} < r$$

A correct match to a distinctive keypoint has a significantly smaller distance than the
second-best; a keypoint on a periodic structure has many near-equally-distant candidates
so the ratio approaches 1 and the match is rejected.  Deployed value: $r = 0.65$.

#### 4.3.2 Mutual Cross-Check

The mutual cross-check requires the match to be reciprocally best in both query directions
and to pass the ratio test in both.  This is strictly stronger than OpenCV's
`crossCheck=True` flag, which enforces only nearest-neighbour reciprocity without the ratio
test:

$$\text{accept}(q, m) \iff \text{RatioTest}(q \to m) \;\wedge\; \text{RatioTest}(m \to q)$$

On periodic scenes, many descriptors that pass the one-way ratio test fail the reverse
because the best reverse match goes to a different instance of the same repeated feature,
not back to $q$.

#### 4.3.3 MAD Displacement Pre-Filter

Let $\boldsymbol{\Delta}_i = \mathbf{p}_i^{(r)} - \mathbf{p}_i^{(c)}$ be the displacement
vector of match $i$.  For a rigid in-plane motion all true inlier displacements cluster
tightly around the true motion $(t_x, t_y)$.  A wrong-cell match differs by an integer
multiple of the grid cell size — it lies in a distinct satellite cluster.

$$\boldsymbol{\mu} = \mathrm{median}_i(\boldsymbol{\Delta}_i), \qquad
\delta_i = \|\boldsymbol{\Delta}_i - \boldsymbol{\mu}\|_2$$

$$\text{MAD} = \mathrm{median}_i(\delta_i) + \varepsilon, \qquad
\text{keep}_i \iff \delta_i < \kappa \cdot \text{MAD}$$

with $\kappa = 4.0$ ($\approx 4\sigma$ equivalent for a normal distribution).  The
satellite clusters lie at displacement magnitudes of $\geq 1$ grid cell ($\sim 150$ px at
0.5× processing scale) from $\boldsymbol{\mu}$ — many MAD units away, removed cleanly
before RANSAC ever sees them.

#### 4.3.4 RANSAC Similarity Estimation

RANSAC iteratively fits a similarity model to random minimal samples and scores each
hypothesis by its inlier count.  For a similarity transform the minimal sample size is
**2 point pairs** (4 scalar constraints for 4 unknowns: $a, b, t_x, t_y$).

A match $i$ is an inlier under hypothesis $\mathbf{H}^{(k)}_{\text{sim}}$ iff:

$$\left\|\mathbf{H}^{(k)}_{\text{sim}}\,\tilde{\mathbf{p}}^{(c)}_i -
\tilde{\mathbf{p}}^{(r)}_i\right\|_2 \leq \tau, \qquad \tau = 3.0\text{ px}$$

The number of iterations required to guarantee (with probability $p_{\text{conf}}$) that
at least one all-inlier 2-point sample is drawn is:

$$N_{\text{iter}} = \frac{\log(1 - p_{\text{conf}})}{\log\!\left(1 - \epsilon_{\text{in}}^{2}\right)}$$

With $p_{\text{conf}} = 0.999$ and $\epsilon_{\text{in}} = 0.5$ this gives $N \approx 34$;
the 5000-iteration cap provides headroom at very low inlier ratios.  The winning hypothesis
is refit on all its inliers (`refineIters=50`) to obtain a least-squares optimal solution.

### 4.4 Candidate Matching and the Temporal Reference Pool

For each new frame, the best alignment reference is selected from a **two-level pool**:

1. **Recent ring buffer** (last $L$ successfully placed frames, default $L=8$) — provides
   temporal continuity across consecutive flight frames.
2. **Keyframe archive** (every $k$-th placed frame is cached, default $k=10$, max 20 entries)
   — provides long-range re-localization anchors, enabling the stitcher to recover when a
   later sweep pass overlaps an earlier one.

All candidates are matched in parallel.  The candidate producing the highest RANSAC inlier
count (subject to a composed-homography validation check) is selected as the reference for
this frame.

### 4.5 Canvas Management and ROI Warping

The canvas is **pre-allocated** as a single fixed array (default 8000×8000 px, ~183 MB)
at initialization.  The first frame is placed at the center; subsequent frames are always
expressed in the coordinate system of this fixed canvas.

Each frame is warped into only its **bounding-box footprint** on the canvas (frame-sized,
not canvas-sized) using a sub-homography shifted to the ROI origin.  This reduces the
transient warp buffer from ~140 MB (full canvas) to ~6 MB (frame footprint), enabling
continuous processing without memory spikes.

### 4.6 Blending

**Coverage mask**: a boolean array maintained alongside the canvas marks every pixel that
has been painted, regardless of color value.  This correctly handles genuinely black scene
pixels (obstacles, floor) that would otherwise be misidentified as unpainted canvas.

In overlap regions, the **Laplacian pyramid blend** (default) decomposes both the existing
canvas ROI and the warped new frame into multi-scale frequency bands, blends each band
with an alpha map derived from a dual distance transform, then reconstructs:

$$\alpha(p) = \frac{d_{\text{new}}(p)}{d_{\text{new}}(p) + d_{\text{old}}(p) + \epsilon}$$

where $d_{\text{new}}$ and $d_{\text{old}}$ are the L2 distance transforms of each
footprint mask.  This seats the seam at the **iso-depth contour** of both footprints,
giving each source equal weight at its own boundary.  The multi-band decomposition
prevents visible seams at high spatial frequencies (obstacle edges, marker corners).

A 4-level pyramid is used by default; blending operates only within the overlap ROI, not
the full canvas.

### 4.7 Pose Graph Optimization (Available, Not Active in Deployed Baseline)

The incremental stitcher accumulates errors: each frame is placed by composing a pairwise
estimate onto a reference, and errors compound along the chain.  A **pose graph optimizer**
is implemented but disabled in the deployed baseline (confirmed by sweep v2 to hurt rather
than help under current arena conditions).

The optimizer parametrizes each frame pose as a similarity $(a, b, t_x, t_y)$.  Every
constraint — sequential odometry edge, loop-closure edge, ArUco marker sighting — becomes
a linear residual in these parameters, reducing global optimization to a single weighted
least-squares system re-solved with Huber IRLS for robustness.  After the video ends,
corrected poses replace the online poses and the map is re-rendered from a disk cache of
all placed frames without reloading the video.

---

## 5. Stage 3 — Obstacle Transfer

The stitched composite image is a noisy aerial reconstruction using color-replaced pixels
to encode semantic classes (pink = obstacle, green = wall).  Direct occupancy rasterization
on this image would propagate stitching distortions into the grid.  A dedicated transfer
pipeline projects extracted obstacle shapes onto the clean `background.png` template,
separating geometric reconstruction from semantic labeling.

**Pipeline:**

```
Input stitched image
    │
    ▼
(5.1) Blue-grid de-warp
      Detect blue tape lines via HSV + probabilistic Hough transform;
      identify horizontal/vertical grid directions; apply affine correction
      so tape lines become axis-aligned
    │
    ▼
(5.2) Crop + wall masking
      Crop to the grid boundary (columns/rows with ≥ N blue pixels).
      Build the green-wall mask; take its convex hull with dilation;
      zero out everything outside. Eliminates noisy pink border artifacts
      that appear outside the wall in the stitched image
    │
    ▼
(5.3) Color masks + morphological closing
      Extract clean pink (obstacle) and green (wall) binary masks.
      Apply morphological closing to bridge intra-blob gaps from
      feather-blending seams
    │
    ▼
(5.4) Blob extraction
      Connected-components analysis on the cleaned pink mask.
      Blobs outside a configurable area band are discarded.
      No shape classification — boxes may appear as joint or
      irregular shapes whose geometry resists single-object heuristics
    │
    ▼
(5.5) Projection onto background.png
      Two modes:
        bbox  — normalize cleaned-image extent → inner wall bbox of
                background.png; scale each obstacle contour accordingly
        grid  — detect blue grid intersections in both images;
                build piecewise-linear cell-to-cell mapping per contour point
      The bbox mode is the primary result; grid mode provides an
      independent projection for consistency scoring (§6)
    │
    ▼
Composited output: background.png + drawn obstacle contours
```

---

## 6. Stage 4 — Consistency Scoring

No ground-truth obstacle positions are available at runtime.  Per-obstacle confidence is
approximated from three algorithmically independent proxies, each in $[0, 1]$:

**Proxy A — bbox↔grid centroid agreement:**
The two projection modes (bbox and grid) of §5.5 produce independent position estimates
for the same source blob.  The centroid displacement $d$ between the two projections,
normalized by the detected grid cell width $w_c$, is converted to a confidence score via a
Gaussian:

$$s_A = \exp\!\left(-\frac{(d / w_c)^2}{\tau_A^2}\right)$$

A displacement below ~0.1 cells gives $s_A \approx 1$; above $\tau_A = 0.5$ cells it
approaches 0.

**Proxy B — bbox↔grid area-ratio agreement:**
The ratio of obstacle areas between the two projections:

$$s_B = \frac{\min(A_{\text{bbox}},\, A_{\text{grid}})}{\max(A_{\text{bbox}},\, A_{\text{grid}})}$$

**Proxy C — perturbation stability:**
The transfer pipeline is re-run with morphological closing iterations offset by $\pm 1$
from the nominal value.  For each obstacle, the centroid standard deviation (normalized
by $w_c$) and the area coefficient of variation are combined into a stability measure:

$$\text{instability} = \frac{\sigma_{\text{centroid}}}{w_c} + \text{CoV}(A)$$

$$s_C = \exp\!\left(-\frac{\text{instability}^2}{\tau_C^2}\right)$$

All four pipeline passes (bbox, grid, perturb$-$, perturb$+$) run **in parallel**.  The
per-obstacle confidence is their weighted combination:

$$c = \frac{w_A s_A + w_B s_B + w_C s_C}{w_A + w_B + w_C}$$

with default weights $w_A = 0.45$, $w_B = 0.25$, $w_C = 0.30$.

> **Figure placeholder:** scatter plot of $c$ distribution across sweep runs — $x$: mean obstacle
> confidence, $y$: SSIM vs. ground truth.  Expected: strong positive correlation confirming
> that the proxy is a valid quality signal.

---

## 7. Stage 5 — Occupancy Rasterization

### 7.1 Pixel Class Semantics

The composited BGR image is analyzed by HSV color to assign occupancy probabilities to each
pixel before grid resampling:

| Region | Detection | Occupancy value |
|---|---|---|
| Black floor + blue grid lines (interior) | Default (not wall, not unknown) | 10 (10%) |
| Obstacle core (drawn shape) | Contour from consistency stage | 90 (90%) |
| Brown perimeter wall | HSV hue 5–25, S 50–180 | 100 (100%) |
| White exterior corners | HSV V ≥ 230, S ≤ 25 | −1 (unknown) |

### 7.2 Confidence-Weighted Halo Thickening

Each obstacle is drawn with an **uncertainty halo** whose thickness is determined by the
per-obstacle consistency score $c \in [0, 1]$:

$$t(c) = t_{\text{base}} \cdot e^{-\lambda c}$$

At $c = 0$ (no confidence) the halo is $t_{\text{base}}$ pixels wide; at $c = 1$ the halo
reduces to $\approx 5\%$ of $t_{\text{base}}$ (with the default $\lambda = 3.0$).

Concentric rings of thickness $\Delta_{\text{ring}}$ are drawn from the outermost inward.
Each ring's occupancy interpolates linearly between the obstacle core value (90%) and the
background floor value (10%):

$$\text{occ}_r = 90 + (10 - 90) \cdot \frac{r}{n_{\text{rings}}}$$

where $r = n_{\text{rings}}$ is the outermost ring and $r = 1$ the innermost.  A ring
cell's occupancy is only raised, never lowered — so neighboring obstacle halos cannot pull
each other down, and high-confidence obstacles are drawn last to prevent their cores from
being covered by low-confidence halos.

### 7.3 Metric Scaling and nav2 Convention

The arena bounding box (inner edge of the brown wall) is detected in the composited image
and mapped to the physical arena dimensions (4.0 × 4.0 m).  The occupancy array is
resampled to the nav2 grid resolution (default 5 cm/cell → 80×80 cells).

The coordinate convention follows nav2 standard: the map origin is at the **bottom-left**
of the arena bounding box, $+x$ right, $+y$ up.  Since the image has a top-left origin
($+y$ down), the occupancy array is flipped vertically before being flattened into the
row-major `OccupancyGrid.data` buffer.

A 2-cell wide wall border is re-stamped after resampling to guarantee a clean, gapless
boundary in the published map regardless of sub-cell quantization from arena-bbox
misalignment.

---

## 8. Marker Localization in the Map

The ArUco markers (goal and AMR fiducials) are identified in the stitched map as solid-
color patches: the goal marker is recolored to solid red (BGR 0,0,255), the AMR marker to
solid cyan (BGR 255,255,0) during the stitching stage.

After the transfer pipeline projects these colored patches onto `background.png`, their
centroids are detected by color thresholding and reported in the same metric frame as the
occupancy grid:

$$x_{\text{m}} = \frac{c_x}{W_{\text{src}}} \cdot w_{\text{arena}}, \quad
y_{\text{m}} = \left(1 - \frac{c_y}{H_{\text{src}}}\right) \cdot h_{\text{arena}}$$

where $(c_x, c_y)$ is the contour centroid and $(W_{\text{src}}, H_{\text{src}})$ is the
cleaned-image extent.  These positions are returned in the `BuildArenaMap` action result
as `goal_marker_position` and `amr_marker_position` and consumed by the mission
orchestrator at stage 04.b for world-frame localization.

---

## 9. Action Interface

The ROS 2 action `BuildArenaMap` provides asynchronous execution with per-stage progress
feedback.  The goal message carries only the video path; all tunable parameters are exposed
as ROS 2 parameters on the action server node and updated between runs without node
restart.

**Feedback stages** (emitted progressively):
```
"stitching"    →  drone_map_grid_gen online/offline pass running
"transferring" →  obstacle transfer running
"consistency"  →  multi-pass consistency scoring running
"occupancy"    →  OccupancyGrid rasterization and marker detection
"done"         →  result imminent
```

**Result fields:**
```
map                    nav_msgs/OccupancyGrid    final probabilistic grid
success                bool
n_obstacles            uint32                    number of accepted obstacles
mean_consistency       float32                   mean per-obstacle confidence
goal_marker_position   geometry_msgs/Point       goal ArUco in world-frame metres
amr_marker_position    geometry_msgs/Point       AMR fiducial in world-frame metres
feature_names[]        string[]                  46-element diagnostic feature names
feature_values[]       float64[]                 corresponding values (for quality gate)
```

The diagnostic feature vector enables the mission orchestrator's `MapQualityClassifier`
to evaluate map acceptability independently of human inspection.

---

## 10. Parameter Sweep and Classifier Training

### 10.1 Sweep Methodology

A systematic parameter sweep was conducted to identify the optimal stitching configuration
for this arena.  The sweep is organized into **iterative versions** (v1 → v2 → v3), each
building on the findings of the previous.

**v3 sweep** covers 335 configurations across 7 parameter groups:

| Group | Dimension | Runs |
|---|---|---|
| A | match_ratio × mad_factor × lookback (dense 3-way, 6×5×5) | 150 |
| B | keyframe_interval × lookback (new dimension, 2×5×6) | 60 |
| C | SuperPoint + ratio_test (GPU backend, 5×3×3) | 45 |
| D | Extraction parameters (fps, movement, blur, artifact thresholds) | 22 |
| E | processing_scale × match_ratio (interaction) | 18 |
| F | feature_exclude_dilate_px × lookback | 20 |
| G | min_keypoint_bins + min_inliers (1-way and 2-way) | 20 |

Each configuration runs the complete 5-stage pipeline on a reference scan video.  Outputs
are `n_obstacles`, `mean_consistency`, `runtime_s`, and the full 46-feature diagnostic
vector.  SSIM against a reference occupancy image is optionally computed.

### 10.2 Key Findings

**v2 finding: lookback is a stability threshold, not a quality knob.**
Group L in v2 (3-way factorial) was the only group that found more than 2 obstacles.
Within it, the transition from `lookback = 8` to `lookback = 16` at the same
`(match_ratio=0.70, mad_factor=4.0)` flipped the result from 0 obstacles to 8 — a binary
outcome, not a gradual improvement.  This identified `lookback` as a minimum threshold
for matching stability rather than a continuous quality parameter.

**v2 finding: all alignment extensions hurt.**
Grid intersection refinement, pose graph optimization, and fiducial loop closure were each
individually evaluated and all reduced map quality under the arena's lighting and grid
conditions.  The deployed baseline disables all three (`use_grid_intersections=False`,
`use_pose_graph=False`, `use_fiducials=False`).

**v2 finding: blue-tape feature exclusion is essential.**
The `feature_exclude_hsv=blue_tape` setting was confirmed as non-negotiable — without it
the RANSAC estimator is dominated by inter-cell false matches and produces degenerate maps.

**Deployed baseline (v2 rank-1, configuration L03):**

| Parameter | Value |
|---|---|
| `feature_extractor` | sift |
| `match_ratio` | 0.65 |
| `mad_factor` | 4.0 |
| `lookback` | 8 |
| `keyframe_interval` | 10 |
| `processing_scale` | 0.5 |
| `feature_exclude_hsv` | blue_tape |
| `feature_exclude_dilate_px` | 5 |

> **Figure placeholder:** Group A v3 heat-map — `n_obstacles` or `mean_consistency` as a
> function of (match_ratio, mad_factor) for each lookback value.  Expected: shows the
> minimum lookback threshold and the (mr, mf) interaction clearly.

### 10.3 RandomForest Map Quality Classifier

The 46-element diagnostic feature vector produced by the pipeline is used to train a
RandomForest binary classifier.  The classifier answers one question at mission time:
*Is the map acceptable for navigation?*

Features are organized by pipeline stage:

| Group | Source | Example features |
|---|---|---|
| 1 | Stitcher image stats | Coverage fraction, mean consistency, canvas fill ratio |
| 2–5 | Transfer pipeline | Obstacle count, blob area statistics, bbox–grid agreement |
| 6 | Pose graph / finalize report | RMS residual before/after solve, marker count |

One feature (`inter_marker_distance_norm`) is dropped at training time (no group prefix in
the feature vector schema), leaving **45 active features**.  Missing features are filled
with a sentinel value of $-1.0$ at inference time, which the model treats as a degraded
map — a safe-fail default.

The model is trained in a Jupyter notebook (`sweep/stitching/train_classifier.ipynb`).
For Jetson deployment, the RandomForest is exported to a portable numpy archive
(`forest.npz`) that reproduces scikit-learn's `predict_proba` in pure numpy (no scikit-learn
at inference).  The decision rule used by the mission orchestrator is:

$$\text{map acceptable} \iff P(\text{pass} \mid \mathbf{f}) \geq \tau$$

where $\tau$ is the threshold optimized during training (stored in `threshold.json`).

---

## 11. Memory Architecture

The pipeline was specifically designed for the Jetson Orin Nano's constrained RAM (8 GB
shared CPU/GPU).  Key memory decisions:

| Design choice | Peak saving | Rationale |
|---|---|---|
| Streaming generator (one frame in RAM at a time) | Eliminates full-video frame list | A 30 min scan at 5 fps = 9000 frames × 6 MB/frame = 54 GB if buffered |
| Pre-allocated fixed canvas (8000×8000) | Eliminates realloc spikes | Dynamic realloc held two full canvases (~360 MB) simultaneously |
| ROI-based warping (frame-footprint, not canvas-sized) | ~140 MB → ~6 MB per warp | The critical memory improvement enabling large canvases |
| Producer-consumer prefetch (queue depth 2) | ~10% throughput gain | Decouples decode from stitching; one frame prefetched while previous is stitched |
| Periodic `malloc_trim` every 50 frames | Prevents RSS inflation | Python/numpy allocator fragmentation inflates RSS without actual leaks |
| Keyframe cap (max 20) | ~50 MB bound | Each SIFT keyframe holds ~2.5 MB of descriptors; unbounded growth depletes RAM |

Peak working set with defaults: **~287 MB** (canvas 183 + keyframes 50 + recent buffer 20 + frame transients 34).

---

## 12. Package Documentation Notes

> **Figures to add for final report:**
> - End-to-end pipeline block diagram with data types at each stage boundary
> - Example stitched map with and without feature-exclusion masking (shows the improvement)
> - Side-by-side: raw stitched composite vs. after transfer onto background.png
> - Occupancy grid visualization with confidence halos (heatmap of occupancy value)
> - Sweep Group A v3 heat-map: n_obstacles vs. (match_ratio × mad_factor) per lookback
> - Classifier ROC curve and precision-recall curve from training notebook
> - Memory profile trace: RSS vs. frames processed (shows flat profile with pre-allocated canvas)
