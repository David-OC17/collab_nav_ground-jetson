# `trajectory_planner` — A* Path Planning and Spline Trajectory Execution

## 1. Role in the System

The trajectory planner solves two coupled problems: (a) finding a collision-free discrete path on the fused occupancy grid from the current robot pose to the goal, and (b) converting that discrete path into a smooth, time-parameterized reference trajectory for the motion controller. A third component — the feedback linearization controller in `amr_bringup` — closes the loop by tracking the reference.

The three-node pipeline is:

```
/fusion/map  ──────────────────────────────────────────────┐
/aruco/goal/pose  ─────────────────────────────────────────┤
TF world→base_footprint  ──────────────────────────────────┤
                                                           ▼
                                              AStarPlanner2  (Jetson)
                                                           │
                                      /trajectory_planner2/path  (nav_msgs/Path)
                                                           │
                                                           ▼
                                              SplineFollower  (Jetson)
                                                           │
                                              /amr/reference  (nav_msgs/Odometry)
                                                           │
                                                           ▼
                                              ControllerNode  (RDK X3)
                                                           │
                                              /amr/cmd_vel  (geometry_msgs/Twist)
                                                           │
                                                           ▼
                                                   motor driver
```

The AEB node (`emergency_stop`) monitors `/amr/emergency_stop` in the motion driver and overrides any command with zero velocity regardless of what the spline follower or controller output.

---

## 2. Stage 1 — Obstacle Inflation

Raw occupancy grid cells carry values in $[-1, 100]$: $-1$ = unknown, $0$ = free, $100$ = fully occupied. Before planning, every lethal cell (value $\geq 90$) is expanded by an **inflation layer** to encode the robot's footprint and a safety margin.

### 2.1 Inflation LUT

The inflation is precomputed into a lookup table (LUT) indexed by offset $(di, dj)$ from each obstacle cell:

```
Procedure BuildInflationLUT(inflation_radius, robot_radius, resolution, cost_scaling):
    r        ← ceil(inflation_radius / resolution)
    inscribed ← ceil(robot_radius / resolution)   # cells
    LUT      ← {}

    for di in [-r, +r]:
        for dj in [-r, +r]:
            dist_cells ← sqrt(di² + dj²)
            dist_m     ← dist_cells × resolution
            if dist_m > inflation_radius or dist_cells = 0:
                continue
            if dist_cells ≤ inscribed:
                cost ← 99.0          # inscribed zone — near-lethal
            else:
                cost ← max(99 · exp(-k · (dist_m - r_robot)), 1.0)
            LUT[(di, dj)] ← cost
    return LUT
```

**Cost profile.** Let $r_\text{robot} = 0.20$ m be the robot inscribed radius, $r_\text{inf} = 0.20$ m the inflation radius, $k = 3.5$ the cost-scaling factor:

$$
c(d) = \begin{cases}
99 & d \leq r_\text{robot} \\
\max\!\left(99 \cdot e^{-k(d - r_\text{robot})},\; 1\right) & r_\text{robot} < d \leq r_\text{inf}
\end{cases}
$$

The inscribed zone ($d \leq r_\text{robot}$) gets cost 99 — a path passing through it would cause a collision with the robot's physical footprint. The exponential decay beyond the inscribed radius penalises cells near obstacles without marking them lethal, biasing A* toward the centre of free corridors.

### 2.2 Inflation Application

```
Procedure InflateMap(raw_map, LUT):
    inflated ← float32 copy of raw_map
    for each cell (cj, ci) with raw_map[cj, ci] ≥ 90:        # lethal
        for (di, dj), cost in LUT:
            ni, nj ← ci + di, cj + dj
            if in bounds and inflated[nj, ni] < cost:
                inflated[nj, ni] ← cost
    return inflated
```

The LUT is built once on the first map message and reused for all subsequent maps (assuming constant resolution).

---

## 3. Stage 2 — A* Path Search

### 3.1 Graph Definition

The planning graph is the 8-connected grid. Nodes are cell coordinates $(c_i, c_j)$. An edge exists from $(c_i, c_j)$ to each of its 8 neighbours provided the neighbour is:
- within map bounds, and
- not lethal: $\text{inflated}[c_j, c_i] < 90$.

**Edge weight.** Each edge carries a base movement cost (Euclidean distance in cell units) plus an occupancy cost proportional to the inflated value of the destination cell:

$$
w(n \to n') = d_\text{move} + c_\text{occ}(n')
$$

$$
d_\text{move} = \begin{cases} 1 & \text{straight} \\ \sqrt{2} & \text{diagonal} \end{cases}
\qquad
c_\text{occ}(n') = \begin{cases} 0.5 & v_{n'} < 0 \text{ (unknown)} \\ \frac{v_{n'}}{100} \times 5 & \text{otherwise} \end{cases}
$$

Unknown cells incur a small 0.5 penalty (exploration discouraged but not forbidden). A free cell ($v = 0$) costs 0, and a near-lethal cell ($v = 85$) costs 4.25 — making A* prefer paths through well-mapped free space.

### 3.2 Octile Heuristic

The heuristic used is the **octile distance**, which is admissible and consistent for 8-connected grids with the above move costs:

$$
h(n, g) = \Delta x + \Delta y + (\sqrt{2} - 2)\min(\Delta x, \Delta y)
$$

where $\Delta x = |c_i - g_i|$, $\Delta y = |c_j - g_j|$. This equals $\max(\Delta x, \Delta y) + (\sqrt{2}-1)\min(\Delta x, \Delta y)$, the exact cost of the optimal unconstrained diagonal path.

Admissibility: $h(n,g)$ never overestimates the true cost because (i) the base edge cost is at least $d_\text{move}$ and (ii) occupancy costs are non-negative, so $h \leq g^*$.

### 3.3 A* Algorithm

```
Procedure AStar(start (si, sj), goal (gi, gj), inflated_map):
    open  ← min-heap, push (f=0, g=0, si, sj)
    G     ← {(si,sj): 0.0}
    prev  ← {}

    while open not empty:
        f, g, ci, cj ← pop(open)

        if (ci, cj) = (gi, gj):
            return ReconstructPath(prev, ci, cj)

        if g > G[(ci, cj)]:           # stale heap entry — skip
            continue

        for each neighbour (ni, nj) with move_cost d_move:
            if out of bounds or lethal: continue
            g' ← g + d_move + occ_cost(ni, nj)
            if g' < G.get((ni, nj), ∞):
                G[(ni, nj)] ← g'
                prev[(ni, nj)] ← (ci, cj)
                push(open, (g' + h(ni, nj), g', ni, nj))

    return None   # no path
```

The closed set is implicit: stale open-set entries are filtered by the `g > G[node]` guard. This is the standard lazy deletion technique — correct because $g$ only monotonically decreases as better paths are found.

---

## 4. Stage 3 — Replanning Logic

The planner maintains an active path and triggers a replan on four conditions:

| Trigger | Condition |
|---|---|
| **First plan** | Map, goal, and TF all become available and no path exists |
| **New goal** | Goal displacement $\|p_\text{new} - p_\text{last}\| > 0.30$ m |
| **Path blocked** | Any waypoint on the current path has inflated cost $> 80$ in the new map |
| **Map diff** | Global change ratio $> 0.05$ OR proximity-weighted score $> 5.0$ |

**Global change ratio.** Let $N_\text{new}$ be the number of cells that transitioned from free/unknown to occupied between the previous and current raw map:

$$
\rho_\text{global} = \frac{N_\text{new}}{W \times H}
$$

**Proximity-weighted score.** For each newly occupied cell $i$ at world position $\mathbf{q}_i$, with $d_i = \min_{p \in \text{path}} \|\mathbf{q}_i - p\|$:

$$
\sigma_\text{prox} = \sum_{i} e^{-d_i / r_\text{prox}}
$$

with $r_\text{prox} = 2.0$ m. This down-weights obstacle changes far from the path and amplifies changes directly in the planned corridor.

Either $\rho_\text{global} > 0.05$ or $\sigma_\text{prox} > 5.0$ triggers a replan; the path-blocked check (direct lethal test) takes priority and is evaluated first.

**Rate limiting.** Replans are suppressed unless at least 3.0 s have elapsed since the last one, preventing rapid oscillation in dynamic environments.

**Goal failure notification.** If the goal cell is lethal at plan time, the planner publishes to `/astar/goal_failed` so `frontier_explorer` can blacklist the position and select an alternative.

---

## 5. Stage 4 — Cubic Spline Trajectory

### 5.1 Arc-Length Parameterization

The raw A* path is a sequence of grid-cell centres $\{(x_k, y_k)\}_{k=0}^{N}$. Fitting a spline directly against a uniform index parameter would cause uneven spatial speed; instead, the parameter is **arc length** $s$:

$$
s_0 = 0, \qquad s_k = \sum_{j=0}^{k-1} \|p_{j+1} - p_j\|_2, \qquad L = s_N
$$

Zero-length segments (duplicate waypoints) are clamped to $10^{-9}$ m before accumulation, and duplicate $s$ values are de-duplicated before spline fitting.

### 5.2 Spline Fitting

Two independent **natural cubic splines** are fitted: $x(s)$ and $y(s)$, using `scipy.interpolate.CubicSpline` with `bc_type='natural'` (second derivative zero at both endpoints):

$$
x''(0) = x''(L) = 0, \quad y''(0) = y''(L) = 0
$$

Natural boundary conditions prevent artificial curvature at the path endpoints. The resulting spline is $C^2$ continuous throughout — position, velocity, and acceleration are all continuous, which is a prerequisite for bounded curvature commands to the controller.

### 5.3 Trapezoidal Speed Profile

The spline follower imposes a **trapezoidal velocity profile** over arc length:

$$
d_\text{ramp} = \frac{v_{\max}^2}{2 a_{\max}}
$$

With $v_{\max} = 0.30$ m/s and $a_{\max} = 0.20$ m/s², $d_\text{ramp} = 0.225$ m.

**Full trapezoid** ($2 d_\text{ramp} < L$):

$$
v(s) = \begin{cases}
\sqrt{2 a_{\max} s} & 0 \leq s \leq d_\text{ramp} \\
v_{\max} & d_\text{ramp} < s < L - d_\text{ramp} \\
\sqrt{2 a_{\max}(L - s)} & L - d_\text{ramp} \leq s \leq L
\end{cases}
$$

**Triangular profile** ($2 d_\text{ramp} \geq L$, path too short to reach $v_{\max}$):

$$
v_\text{peak} = \sqrt{a_{\max} L}, \qquad
v(s) = \begin{cases}
\sqrt{2 a_{\max} s} & 0 \leq s \leq L/2 \\
\sqrt{2 a_{\max}(L - s)} & L/2 < s \leq L
\end{cases}
$$

The profile guarantees $v(0) = v(L) = 0$ (starts and stops at rest) and $|dv/ds| \leq a_{\max}$ at every point.

### 5.4 Curvature and Angular Velocity

At each arc-length position $s$, the tangent direction and curvature are evaluated from the spline derivatives:

$$
\psi(s) = \text{atan2}\!\left(\frac{dy}{ds}, \frac{dx}{ds}\right)
$$

$$
\kappa(s) = \frac{\dot{x}\ddot{y} - \dot{y}\ddot{x}}{\left(\dot{x}^2 + \dot{y}^2\right)^{3/2}}
$$

where dots denote derivatives with respect to $s$. The angular velocity command follows directly from the kinematic relationship $\omega = \kappa v$:

$$
\omega(s) = \kappa(s) \cdot v(s)
$$

### 5.5 Reference Output and Arc-Length Integration

At each 20 Hz tick, the follower:

1. Evaluates $x(s), y(s), \psi(s), \kappa(s)$ at the current $s$.
2. Computes $v(s)$ from the trapezoidal profile.
3. Rotates world-frame velocity $(v \cos\psi, v \sin\psi)$ into robot frame:

$$
\begin{bmatrix} v_x^\text{robot} \\ v_y^\text{robot} \end{bmatrix}
=
\begin{bmatrix} \cos\psi & \sin\psi \\ -\sin\psi & \cos\psi \end{bmatrix}
\begin{bmatrix} v \cos\psi \\ v \sin\psi \end{bmatrix}
=
\begin{bmatrix} v \\ 0 \end{bmatrix}
$$

Because yaw is always aligned with the tangent, $v_y^\text{robot} = 0$ exactly (no lateral command). The reference is published as `nav_msgs/Odometry` on `/amr/reference`.

4. Advances the arc-length: $s_{t+1} = s_t + v(s_t) \cdot \Delta t$ (Euler integration).

Goal termination fires when $L - s < 0.10$ m.

---

## 6. Stage 5 — Feedback Linearization Controller

The controller (`amr_bringup/controller_node.py`, Yahboom RDK X3) closes the position loop. It runs at **100 Hz**.

### 6.1 Differential Drive Kinematics

The differential-drive kinematic model in the world frame is:

$$
\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega
$$

This system is **not feedback linearizable** at the robot center because the input $(v, \omega)$ cannot decouple $\dot{x}$ and $\dot{y}$ independently — the heading $\theta$ creates a nonholonomic constraint.

### 6.2 Virtual Look-Ahead Point

The standard remedy is to control a **virtual point** located a distance $R$ ahead of the robot center along its heading:

$$
\mathbf{p}_R = \begin{bmatrix} x + R\cos\theta \\ y + R\sin\theta \end{bmatrix}
$$

The time derivatives of the virtual point are:

$$
\dot{x}_R = v\cos\theta - R\omega\sin\theta
\qquad
\dot{y}_R = v\sin\theta + R\omega\cos\theta
$$

In matrix form:

$$
\underbrace{\begin{bmatrix} \dot{x}_R \\ \dot{y}_R \end{bmatrix}}_{\mathbf{u}}
=
\underbrace{\begin{bmatrix} \cos\theta & -R\sin\theta \\ \sin\theta & R\cos\theta \end{bmatrix}}_{\mathbf{T}(\theta)}
\begin{bmatrix} v \\ \omega \end{bmatrix}
$$

$\det(\mathbf{T}) = R > 0$ for all $\theta$, so $\mathbf{T}$ is always invertible.

### 6.3 Control Law

The virtual control input $\mathbf{u} = [u_1, u_2]^T$ is chosen as a **proportional + feedforward** law:

$$
u_1 = k_{px}(x_\text{ref} - x_R) + \dot{x}_\text{ref}
\qquad
u_2 = k_{py}(y_\text{ref} - y_R) + \dot{y}_\text{ref}
$$

where $(x_\text{ref}, y_\text{ref})$ and $(\dot{x}_\text{ref}, \dot{y}_\text{ref})$ come from the `/amr/reference` Odometry. Substituting into the virtual point error $e_i = p_{R,i} - p_{\text{ref},i}$:

$$
\dot{e}_1 = -k_{px} e_1, \qquad \dot{e}_2 = -k_{py} e_2
$$

The two axes decouple and each converges exponentially with time constants $1/k_{px}$ and $1/k_{py}$. With $k_{px} = 1.0$, $k_{py} = 2.0$, the y-axis converges twice as fast (lateral error is penalized more strongly).

### 6.4 Actual Commands

Inverting $\mathbf{T}$:

$$
\begin{bmatrix} v \\ \omega \end{bmatrix}
=
\mathbf{T}^{-1}(\theta) \mathbf{u}
=
\frac{1}{R}
\begin{bmatrix} R\cos\theta & R\sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} u_1 \\ u_2 \end{bmatrix}
$$

$$
v = u_1 \cos\theta + u_2 \sin\theta
\qquad
\omega = \frac{-u_1 \sin\theta + u_2 \cos\theta}{R}
$$

A **deadband** of 0.15 m/s is added to the magnitude of $v$ to overcome static friction at motor startup. No deadband is applied to $\omega$.

The robot pose $(\hat{x}, \hat{y}, \hat{\theta})$ is read from the TF tree (world → base_footprint), which is maintained by the EKF (`ekf_amr`) on the RDK X3.

---

## 7. Parameters

### AStarPlanner2

| Parameter | Value | Meaning |
|---|---|---|
| `inflation_radius` | 0.20 m | Outer radius of inflation zone |
| `robot_radius` | 0.20 m | Inscribed circle radius (zone 99 cost) |
| `cost_scaling` | 3.5 | Exponential decay factor $k$ |
| `collision_cost_threshold` | 80.0 | Inflated cost above which a waypoint is "blocked" |
| `global_change_threshold` | 0.05 | New-obstacle ratio for map-diff replan |
| `path_proximity_threshold` | 5.0 | Proximity score for map-diff replan |
| `path_proximity_radius` | 2.0 m | Decay length for proximity score |
| `goal_change_threshold` | 0.30 m | Min goal displacement to accept new goal |
| `min_replan_interval_sec` | 3.0 s | Rate limiter between replans |

### SplineFollower

| Parameter | Value | Meaning |
|---|---|---|
| `max_speed` | 0.30 m/s | Cruise speed $v_{\max}$ |
| `max_accel` | 0.20 m/s² | Ramp acceleration $a_{\max}$ |
| `goal_tolerance` | 0.10 m | Arc-length remaining to declare goal reached |
| `update_rate` | 20 Hz | Reference publish rate |

### ControllerNode (`amr_bringup`)

| Parameter | Value | Meaning |
|---|---|---|
| `kpx` | 1.0 | Proportional gain, x-axis of virtual point |
| `kpy` | 2.0 | Proportional gain, y-axis of virtual point |
| `R` | 0.2 m | Look-ahead distance for virtual point |
| `deadband` | 0.15 m/s | Minimum linear speed applied at actuation |
| `control_rate` | 0.01 s | Controller update period (100 Hz) |

---

## 8. Known Limitations

**Open-loop arc-length integration.** The spline follower advances $s$ by $v \cdot \Delta t$ each tick without measuring actual displacement. If the robot lags the reference (e.g., due to AEB stop followed by resume), the follower continues advancing $s$ and the robot must catch up. The controller's position-tracking term absorbs this, but large lags can cause brief large velocity commands when the stop clears.

**No heading control at start.** If the robot is initially misaligned with the first path segment, the virtual-point controller will rotate the robot through the normal course of position correction rather than first spinning in place. The look-ahead parameter $R = 0.2$ m sets how aggressively the controller steers: smaller $R$ gives tighter tracking but amplifies angular commands.

**Grid resolution.** A* operates on the occupancy grid cell resolution. The fused map at typical resolution (0.05 m/cell) means the raw path has waypoints 0.05–0.07 m apart. After spline fitting, sub-grid smoothness is achieved, but the path is still constrained to corridors that are at least one inflated cell wide.

**Single replanning trigger cycle.** The path-blocked check iterates over all waypoints; if many cells have borderline cost, the trigger may fire on every new map even if the path is geometrically valid. The 3.0 s rate limit prevents this from flooding the planner but may delay reactive replanning in genuinely dynamic scenarios.

**No velocity constraint in A*.** The cost function optimises path length and proximity to obstacles but does not account for the speed profile. A path with many tight bends may force $\kappa(s)$ large enough that $\omega = \kappa v$ saturates the motor commands at cruise speed. Curvature-aware planning (penalizing high-curvature transitions) is not implemented.
