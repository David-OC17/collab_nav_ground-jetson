# `world_mapper` — Probabilistic LiDAR Occupancy Mapper

## 1. Role in the System

`world_mapper` maintains a running 2D occupancy grid of the arena in the `world` frame, built incrementally from OraDAR MS200 LiDAR scans as the AMR moves. It is the ground-level counterpart to the aerial map produced by `arena_map_builder`: where the aerial map provides a bird's-eye view of the arena floor (obstacles, lane markings, ArUco markers), this node provides a continuously-updated obstacle map reflecting what the AMR's own sensor currently sees.

The output `/amr/world_map` (`nav_msgs/OccupancyGrid`) feeds into `map_fusion`, which registers and combines it with the aerial occupancy grid to produce `/fusion/map` — the map consumed by `trajectory_planner`.

The node is a pure listener: it reads `/scan` and the TF tree (world → laser_frame, resolved through the SLAM toolbox's drift-corrected chain) and produces the map. It modifies no state outside its own log-odds array.

---

## 2. Probabilistic Occupancy Model

### 2.1 Binary Bayes Filter

Each cell $(i, j)$ holds a binary random variable $m_{ij} \in \{0, 1\}$ representing occupancy. Given a sequence of sensor measurements $z_{1:t}$, the posterior is maintained recursively. With a uniform prior $p_0 = p(m=1) = 0.5$, the standard result is that the posterior factors into a product of inverse sensor model terms, and the update is most efficiently expressed in **log-odds** form.

Define:

$$
l_t(m) = \log \frac{p(m=1 \mid z_{1:t})}{p(m=0 \mid z_{1:t})}
$$

Then the Bayes update reduces to an additive rule:

$$
l_t(m) = l_{t-1}(m) + \underbrace{\log \frac{p(m=1 \mid z_t)}{p(m=0 \mid z_t)}}_{l_\text{sensor}(z_t)}
$$

where $l_\text{sensor}$ is the **inverse sensor model** evaluated for the current measurement. This is the key computational advantage: instead of a division per cell per scan, each update is a single floating-point addition.

### 2.2 Inverse Sensor Model

For a 2D LiDAR with a measured range $r$ to a beam endpoint, the inverse sensor model assigns:

| Beam cell | $l_\text{sensor}$ | Deployed value | Interpretation |
|---|---|---|---|
| Ray endpoint (real return) | $l_\text{occ}$ | $+0.85$ | $p(\text{occ}\mid\text{hit}) = \sigma(0.85) \approx 0.70$ |
| All cells along ray before endpoint | $l_\text{free}$ | $-0.40$ | $p(\text{occ}\mid\text{pass}) = \sigma(-0.40) \approx 0.40$ |

where $\sigma(x) = 1/(1+e^{-x})$ is the sigmoid function.

The asymmetry $|l_\text{occ}| > |l_\text{free}|$ is intentional: a single confirmed hit takes about 2.1 misses to counteract ($0.85/0.40 \approx 2.1$). This prevents obstacles from disappearing due to transient free-space readings (e.g., a dynamic object moving away briefly) and encodes a conservative safety bias.

### 2.3 Saturation and Clamping

The log-odds is clamped to $[l_\text{min}, l_\text{max}] = [-5.0, +5.0]$ after each update, preventing unbounded growth and ensuring that cells can change state even after long exposure to one observation class:

$$
l_t(m) \leftarrow \text{clip}(l_t(m),\; -5.0,\; +5.0)
$$

The corresponding probability extremes are $\sigma(-5) \approx 0.007$ (near-certain free) and $\sigma(+5) \approx 0.993$ (near-certain occupied). Starting from $l = 0$, saturation requires approximately:

$$
\lceil 5.0 / 0.85 \rceil = 6 \text{ consecutive hits}
\qquad
\lceil 5.0 / 0.40 \rceil = 13 \text{ consecutive misses}
$$

### 2.4 Log-Odds to Occupancy Value

When publishing, the log-odds grid is converted to the ROS `OccupancyGrid` convention:

$$
p_{ij} = \frac{1}{1 + e^{-l_{ij}}}
\qquad
\text{value}_{ij} = \left\lfloor p_{ij} \times 100 \right\rceil
$$

Cells with $l_{ij} = 0.0$ (never observed, prior unchanged) are mapped to $-1$ (ROS unknown convention) rather than 50, since 0.0 log-odds means "no information" not "equally likely occupied or free".

---

## 3. Raycasting via Bresenham's Algorithm

For each valid beam in a scan, the inverse sensor model must be applied to all cells along the ray. The **Bresenham line algorithm** enumerates all integer grid cells on the line from the sensor origin cell $(o_c, o_r)$ to the endpoint cell $(e_c, e_r)$:

```
Procedure Bresenham(x0, y0, x1, y1):
    dx ← |x1 - x0|,  dy ← |y1 - y0|
    sx ← sign(x1 - x0),  sy ← sign(y1 - y0)
    err ← dx - dy
    x, y ← x0, y0

    loop:
        yield (x, y)
        if x = x1 and y = y1: break
        e2 ← 2 * err
        if e2 > -dy:  err ← err - dy;  x ← x + sx
        if e2 <  dx:  err ← err + dx;  y ← y + sy
```

The algorithm produces exactly the cells that a line passes through on a discrete grid, with no floating-point rounding per step. This makes it $O(N)$ per ray where $N$ is the number of cells traversed, and requires only integer arithmetic.

**Ray partitioning.** Given the Bresenham cell sequence $\{c_0, c_1, \ldots, c_k\}$:

- $c_0, \ldots, c_{k-1}$ (all cells before the endpoint): `miss` — apply $l_\text{free}$
- $c_k$ (the endpoint): `hit` — apply $l_\text{occ}$, **unless** the reading is a max-range return

---

## 4. Max-Range and Out-of-Bounds Handling

### 4.1 Max-Range Returns

A reading $r \geq r_\text{max}$ means the sensor ranged out without detecting an obstacle. The endpoint does **not** correspond to a physical surface, so:

- All cells along the ray are marked **free** ($l_\text{free}$), including the endpoint.
- No occupied hit is recorded.

### 4.2 Out-of-Bounds Endpoints

If the ray endpoint lies outside the map, the endpoint cell index is **clamped to the nearest grid boundary cell**. The ray is then traced to the clamped cell, marking all traversed cells free. The clamped endpoint is also treated as max-range (no occupied hit). This ensures that long rays extending beyond the arena still contribute free-space information within the mapped region.

### 4.3 Full Scan Processing

```
Procedure ProcessScan(scan, laser_pose (ox, oy, ψ)):
    (ocol, orow) ← world_to_cell(ox, oy)
    if out of bounds: drop scan

    hits ← [], misses ← []
    for each (r, α) in scan:
        if r is NaN or Inf or r < r_min: skip
        is_max ← (r ≥ r_max)
        rr ← min(r, r_max)
        ex ← ox + rr·cos(ψ + α),   ey ← oy + rr·sin(ψ + α)

        (ecol, erow) ← world_to_cell(ex, ey)
        if out of bounds:
            (ecol, erow) ← clamp_to_grid(...)
            is_max ← True

        cells ← Bresenham(ocol, orow, ecol, erow)
        misses ← misses ∪ cells[0 .. k-1]
        if is_max: misses ← misses ∪ {cells[k]}
        else:        hits ← hits ∪ {cells[k]}

    logodds[misses] += l_free
    logodds[hits]   += l_occ
    logodds ← clip(logodds, l_min, l_max)
```

Updates are batched by scan (accumulating all hits and misses before applying) and applied under a mutex shared with the publish timer.

---

## 5. Pose Source

The node does not run its own localization. It reads the TF transform `world → scan.header.frame_id` at the time of each scan. The expected TF chain is:

```
world ──► slam_map ──► odom ──► base_footprint ──► laser_frame
           ^                ^
           |                slam_toolbox (drift-corrected)
           amr_drone_nav (world↔slam_map alignment)
```

`slam_toolbox` provides a continuously corrected `odom → slam_map` transform that reduces the accumulation of dead-reckoning drift. The `world → slam_map` static alignment is published by `amr_drone_nav`. The world_mapper only calls `tf_buffer.lookup_transform()` — it is indifferent to which node in the chain published each segment.

If the transform is unavailable (timeout 100 ms), the scan is silently dropped with a throttled warning.

---

## 6. Parameters

| Parameter | Deployed Value | Meaning |
|---|---|---|
| `resolution` | 0.05 m/cell | Grid cell size |
| `width_m` | 3.9 m | Map width (78 cells) |
| `height_m` | 3.9 m | Map height (78 cells) |
| `origin_x`, `origin_y` | 0.0, 0.0 m | World-frame position of the bottom-left cell corner |
| `l_occ` | +0.85 | Log-odds increment per hit |
| `l_free` | −0.40 | Log-odds increment per miss (free cell along ray) |
| `l_min` | −5.0 | Log-odds lower clamp |
| `l_max` | +5.0 | Log-odds upper clamp |
| `publish_rate` | 1.0 Hz | Map publication rate |
| `tf_timeout` | 0.10 s | Max wait for TF before dropping scan |

The map covers a 3.9 × 3.9 m area, matching the 4 × 4 m arena with a small margin on each side. The origin is at the world-frame origin; if the AMR starts at (0, 0), all arena cells are within bounds.

---

## 7. Known Limitations

**Fixed map size.** The grid is allocated once at startup. If the AMR's pose drifts outside the mapped region (sensor outside bounds), scans are dropped. No dynamic resizing is implemented.

**No forgetting.** The log-odds model is Markovian: old observations are only "forgotten" by enough opposing observations. A stationary obstacle that is removed from the arena will take ~13 consecutive miss scans from a cell to approach free-space confidence. In practice, as the AMR moves past the cleared area, this converges within a few seconds at 15 Hz scan rate.

**2D only.** The LiDAR scan plane is at a fixed mounting height. Objects entirely above or below the scan plane are invisible. This includes the Tello drone during the scan phase (not a problem since it lands before AMR navigation begins) and overhanging structures.

**No multi-return integration.** Each scan ray contributes exactly one update per cell. Dense reflective surfaces (glass, polished floor) can produce noisy range readings that spread the hit distribution along the beam, inflating obstacle extent.
