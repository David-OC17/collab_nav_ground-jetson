# `fusion` — Map Fusion Node

## 1. Role in the System

The fusion node combines two independently-built occupancy grids — the **drone aerial map** (`arena_map_builder`) and the **AMR live map** (`world_mapper`) — into a single fused grid consumed by `trajectory_planner`. Neither source alone is sufficient:

- The drone map provides a **global prior** of the arena structure, built before the AMR moves. It covers the whole arena in one shot but is subject to aerial stitching artefacts, lighting noise, and cannot detect objects moved after the drone scan.
- The AMR live map provides **dynamic local information** — obstacles the AMR's LiDAR sees in real time — but starts empty and accumulates coverage only as the robot moves.

The fusion result inherits the global layout certainty of the drone map while allowing the AMR sensor to add newly detected obstacles or fill in unknown regions.

---

## 2. Pre-Alignment: Why No Registration Is Needed

Both maps are published in the `world` frame, with identical resolution, dimensions, and origin (guaranteed by construction). `arena_map_builder` projects the stitched aerial image directly into the `world` frame using OptiTrack-derived metric scale and origin. `world_mapper` builds its grid in the same `world` frame by raycasting the LiDAR scan through the TF chain that terminates at `world`. Because the two coordinate systems are already the same frame, cell $(i, j)$ in the drone grid and cell $(i, j)$ in the AMR grid represent the same physical patch of the arena — no SE(2) registration is required before combining them.

### 2.1 Attempted ICP Registration (Deprecated)

A prior implementation (`map_fusion`, removed) performed full SE(2) registration between the two grids:

**Stage 1 — Coarse search.** For each discretised rotation angle $\theta_k$ in $[-\pi, \pi)$:
1. Rotate the SLAM edge point cloud by $\theta_k$.
2. Rasterise the rotated cloud to a binary image at the map resolution.
3. Cross-correlate against the drone edge image via FFT:

$$
\text{score}(\theta_k, \mathbf{t}) = \mathcal{F}^{-1}\!\left[\hat{S}_{\theta_k}^* \cdot \hat{D}\right](\mathbf{t})
$$

   This reduces the translation search from $O(N_t^2)$ to $O(N \log N)$ per rotation. Peak candidates are extracted with greedy non-maximum suppression.

**Stage 2 — ICP refinement.** Each coarse candidate $(\hat{t}_x, \hat{t}_y, \hat{\theta})$ is refined by point-to-point ICP (Besl & McKay):

```
Procedure ICP(source S, target T, init t, params):
    tree ← cKDTree(T)
    for iter = 1 to max_iter:
        S' ← apply_SE2(t, S)
        (dist, idx) ← tree.query(S')
        gated ← {i : dist[i] ≤ max_correspondence}
        if |gated| < 3: break
        keep ← {i ∈ gated : dist[i] ≤ 2 · median(dist[gated])}   # outlier rejection
        Δt ← rigid_fit_2d(S'[keep], T[idx[keep]])                  # closed-form SE(2) fit
        t  ← compose(Δt, t)
        if ||Δt_xy|| < ε and |Δt_θ| < ε: converged; break
    return t, mean inlier residual
```

The per-iteration **2× median outlier gate** rejects gross mismatches before the closed-form SE(2) step, making the solver more robust than standard ICP to partial overlap. Alignment confidence was computed as $c = e^{-r / \sigma}$ where $r$ is the mean inlier residual.

**Why it was abandoned.** The approach requires a sufficiently dense and feature-rich AMR map to produce stable coarse-search peaks. Early in a mission run, the AMR has only scanned a small portion of the arena; the resulting edge cloud is too sparse and non-discriminative, leading to inconsistent alignment across runs. Since OptiTrack-derived pre-alignment is already exact (both sources are in the same `world` frame), the registration added latency and fragility with no accuracy benefit. The current elementwise approach was adopted instead.

---

## 3. Fusion Semantics

The drone map and AMR map carry qualitatively different epistemic content:

| Source | Cell value | Interpretation |
|---|---|---|
| Drone map | $\geq 65$ | Obstacle — high-confidence structural feature |
| Drone map | $-1$ | Unknown — drone view was occluded or not covered |
| Drone map | $< 65$, $\neq -1$ | Free floor — represented as 25, acknowledging aerial-image noise |
| AMR map | any | Live sensor reading, fills in as robot moves |

The drone **free floor value** is fixed at 25 rather than 0. A value of 0 would imply certainty of free space, which is too strong for an aerial stitching pipeline subject to lighting variation, image blur, and grid-projection artefacts. Representing floor as 25 encodes a weak prior: the area is likely free but not guaranteed.

The drone **occupied threshold** is 65, not 50 or 100. Values between 25 and 64 in the drone map are treated as free (noisy but non-obstacle). This gives the aerial pipeline a margin — only strong, consistent dark regions (obstacles) are treated as immutable structure.

### 3.1 Formal Fusion Rule

For each cell $(i, j)$, with $\tau = 65$, $v_f = 25$:

$$
G_F[i,j] = \begin{cases}
100 & G_D[i,j] \geq \tau \quad \text{(drone obstacle — immutable)} \\[4pt]
G_A[i,j] & G_D[i,j] = -1 \quad \text{(drone unknown — defer to AMR)} \\[4pt]
\max(v_f,\; G_A[i,j]) & G_D[i,j] < \tau,\; G_D[i,j] \neq -1,\; G_A[i,j] \neq -1 \\[4pt]
v_f & G_D[i,j] < \tau,\; G_D[i,j] \neq -1,\; G_A[i,j] = -1
\end{cases}
$$

The third rule — $\max(v_f, G_A)$ — is the key design decision: **the AMR can only raise occupancy in drone-free cells, never lower it**. An AMR scan that returns free space where the drone said floor does not change the output below $v_f$. An AMR scan that returns an obstacle (high value) raises the output to that value, registering the new obstacle for the planner.

### 3.2 Pseudocode

```
Procedure Fuse(G_D, G_A, τ=65, v_f=25):
    drone_occ  ← G_D ≥ τ
    drone_unk  ← G_D = -1
    drone_free ← ¬drone_occ ∧ ¬drone_unk

    G_F ← -1 everywhere

    G_F[drone_occ]  ← 100
    G_F[drone_unk]  ← G_A[drone_unk]
    G_F[drone_free] ← where(G_A[drone_free] = -1,
                             v_f,
                             max(v_f, G_A[drone_free]))
    return G_F
```

This is a single vectorised pass over the grid — $O(W \times H)$ per AMR map update.

---

## 4. Node Architecture

```
/drone/map   (OccupancyGrid, TRANSIENT_LOCAL — published once by arena_map_builder)
      │
      ▼  stored as drone prior
MapFusionNode
      ▲
      │  triggers fuse() on every update
/world_map   (OccupancyGrid, ~1 Hz from world_mapper)
      │
      ▼
/fused_map   (OccupancyGrid, published on every AMR update)
      │
      ▼
trajectory_planner (AStarPlanner2)
```

The drone map subscription uses `TRANSIENT_LOCAL` QoS so the node receives the map even if it starts after `arena_map_builder` has already published. The AMR map subscription is best-effort at 1 Hz (the `world_mapper` publish rate). No timer is used — the fused map is published reactively on every incoming AMR map message.

If an AMR map arrives before the drone map is received, the update is silently dropped with a throttled warning. The drone map is the mandatory prior; without it, the fusion output would be meaningless for planning.

---

## 5. Assumptions

- Both grids have **identical dimensions** ($W \times H$) and **resolution** (m/cell). This is guaranteed by configuration: both `arena_map_builder` and `world_mapper` are launched with the same `width_m`, `height_m`, and `resolution` parameters, and both reference the `world` frame origin.
- The drone map is **static** after the stitching phase completes. It is published once and stored. If `arena_map_builder` re-runs and publishes a new map, the fusion node will update its prior on the next drone map callback.
- The fused map inherits the **map metadata** (resolution, origin, dimensions) from the drone map message, not the AMR map. This is safe because both are configured identically.

---

## 6. Known Limitations

**Drone obstacle immutability.** Cells classified as obstacles by the drone cannot be cleared by AMR LiDAR free-space evidence. If the drone map contains a false-positive obstacle (e.g., a shadow or image artefact), the planner will permanently treat that cell as blocked. The threshold $\tau = 65$ mitigates this by requiring a high confidence in the drone map before marking a cell immutable.

**No temporal decay.** The AMR live map (`world_mapper`) uses log-odds with saturation and does not decay over time. An obstacle the AMR saw early in the run will remain in the fused map even if the robot moves away and the obstacle is removed. This is acceptable for the static-obstacle arena but would be a problem in dynamic environments.

**Single prior update.** The drone map is treated as a fixed prior loaded once. Multiple sequential runs with different obstacle layouts would require a full restart of both `arena_map_builder` and `fusion` to update the prior.
