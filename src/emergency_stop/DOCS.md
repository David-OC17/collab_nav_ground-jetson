# `emergency_stop` — Automatic Emergency Braking (AEB)

## 1. Role in the System

The AEB node is a **safety layer that runs independently of the planner**. It monitors the 2D LiDAR scan in real time and publishes a binary stop flag on `/amr/emergency_stop`. The motion driver subscribes to this topic and immediately sends all-zero velocity commands when the flag is high; it may resume normal execution once the flag clears. The node is decoupled from `trajectory_planner`, `mission_orchestrator`, and the EKF — it has no knowledge of the planned path and cannot be disabled by a higher-level failure.

The critical design invariant is:

> **If an obstacle exists within $d_{\min}$ for $n_\text{trig}$ consecutive scans, motion stops. No planning or orchestration failure can prevent this.**

---

## 2. Problem Statement

Let the LiDAR produce a stream of range scans $\{S_t\}$ at frequency $f_s \approx 15$ Hz. Each scan $S_t = \{(r_i, \theta_i)\}_{i=1}^{N}$ contains $N$ range–angle pairs. Define the binary raw detection:

$$
x_t = \mathbf{1}\!\left[\min_{i \in \mathcal{V}(S_t)} r_i < d_{\min}\right]
$$

where $\mathcal{V}(S_t)$ is the set of **valid** readings (finite, within sensor range, outside the blind-spot sector). The AEB problem is to produce a stop signal $\hat{s}_t \in \{0,1\}$ such that:

- **Safety (no false negatives at steady state):** if a real obstacle is within $d_{\min}$ for more than $n_\text{trig}/f_s$ seconds, then $\hat{s}_t = 1$.
- **Availability (low false positive rate):** transient noise spikes of duration $< n_\text{trig}/f_s$ do not set $\hat{s}_t = 1$.
- **Liveness (prompt clearance):** once the obstacle is gone for $n_\text{clear}/f_s$ seconds, $\hat{s}_t$ returns to 0.

The tension between safety and availability is resolved by the **N-consecutive confirmation filter** (§5).

---

## 3. Node Architecture

```
/scan  (LaserScan, ~15 Hz)
    │
    ▼
EmergencyStopNode
    ├── Range preprocessor          §4
    ├── Proximity detector          §4
    ├── Confirmation debounce       §5
    └── Multi-reason state machine  §6
         │
         ▼
/amr/emergency_stop  (Bool, 10 Hz heartbeat + event-driven log)
         │
         ▼
    motion driver  →  zero Twist on HIGH, resume on LOW
```

**Subscriptions**

| Topic | Type | Rate | Purpose |
|---|---|---|---|
| `/scan` | `LaserScan` | ~15 Hz | LiDAR proximity trigger |

**Publications**

| Topic | Type | Rate | Purpose |
|---|---|---|---|
| `/amr/emergency_stop` | `Bool` | 10 Hz | Stop flag to motion driver |

The 10 Hz heartbeat timer publishes the current state unconditionally, ensuring the motion driver receives a fresh signal even if no new scan arrives (e.g., sensor drop-out) — the last known stop state persists.

---

## 4. Range Preprocessing

For each incoming scan, valid readings are extracted as follows:

```
Procedure FilterRanges(scan S, mask_min θ_L, mask_max θ_H, d_min):
    valid ← []
    for i = 0 to |S.ranges| - 1:
        r ← S.ranges[i]
        if not isfinite(r):                     # drop NaN / ±Inf
            continue
        θ ← S.angle_min + i · S.angle_increment
        if θ_L ≤ θ ≤ θ_H:                      # drop blind-spot sector
            continue
        if S.range_min < r < S.range_max:       # drop out-of-sensor-range
            valid.append(r)
    return valid
```

**Blind-spot mask.** The OraDAR MS200 is mounted centrally on the AMR chassis. Its 360° field of view includes the robot's own body in the rear sector, which appears as a permanent wall at ranges $< d_{\min}$. The mask $[\theta_L, \theta_H] = [120°, 240°]$ suppresses this sector (convention: $0°$ = forward, angles measured counter-clockwise). All readings in the rear 120° sector are discarded unconditionally.

The mask intentionally leaves the robot blind to rear obstacles. This is an acceptable trade-off because the AMR operates in a forward-facing navigation mode: the planned trajectory never reverses at speed, so a rear collision is not a credible hazard in normal operation.

**Proximity decision.** After filtering:

$$
x_t = \begin{cases} 1 & \text{if } \text{valid} \neq \emptyset \text{ and } \min(\text{valid}) < d_{\min} \\ 0 & \text{otherwise} \end{cases}
$$

---

## 5. Confirmation Debounce Filter

Raw LiDAR readings exhibit transient outliers from reflective surfaces, glass, moving legs, or RF interference with the time-of-flight circuit. A single bad reading should not halt the robot; conversely, a single good reading after a stop should not immediately release it.

The confirmation filter is an **N-consecutive streak detector** with independent trigger and clear thresholds:

```
State: obstacle_counter ← 0
       clear_counter    ← 0
       stop_active      ← False   (reason: 'lidar_proximity')

On scan callback with raw detection x_t:
    if x_t = 1:
        obstacle_counter ← obstacle_counter + 1
        clear_counter    ← 0
        if obstacle_counter ≥ n_trig:
            SET_STOP('lidar_proximity')       # latch trigger
    else:
        clear_counter    ← clear_counter + 1
        obstacle_counter ← 0
        if clear_counter ≥ n_clear:
            CLEAR_STOP('lidar_proximity')     # latch clear
```

A single non-detection scan resets `obstacle_counter` to zero (and vice versa). This means the filter requires **an unbroken run** of $n_\text{trig}$ positive detections to trigger, and an unbroken run of $n_\text{clear}$ negative detections to clear. It is more conservative than an N-of-M sliding window — any interruption in the streak restarts the count.

**Temporal analysis.**

| Parameter | Value | Duration |
|---|---|---|
| $f_s$ | 15 Hz | — |
| $n_\text{trig}$ | 5 | $\tau_\text{trig} = 5/15 \approx 333$ ms |
| $n_\text{clear}$ | 5 | $\tau_\text{clear} = 5/15 \approx 333$ ms |

The maximum undetected approach time is $\tau_\text{trig}$. During this window at cruise velocity $v$, the robot closes:

$$
\Delta d_\text{debounce} = v \cdot \tau_\text{trig}
$$

The threshold $d_{\min} = 0.30$ m was determined empirically as the minimum distance from which the robot can reliably stop at maximum operational speed before physical contact. This bound implicitly encodes the debounce latency, mechanical braking distance, and localization uncertainty combined.

---

## 6. Multi-Reason State Machine

The stop state is maintained as a **reason set** $\mathcal{R}$:

$$
\hat{s}_t = \mathbf{1}[|\mathcal{R}_t| > 0]
$$

```
Procedure SET_STOP(reason):
    R ← R ∪ {reason}
    stop_active ← (|R| > 0)

Procedure CLEAR_STOP(reason):
    R ← R \ {reason}
    stop_active ← (|R| > 0)
```

Currently only `'lidar_proximity'` is active. The architecture permits additional triggers — speed gate, communication timeout, IMU shock — to independently add and remove reasons without coupling. The stop flag remains high as long as **any** reason is present; all reasons must clear before motion resumes.

A second trigger `_odom_cb` monitors `/amr/ekf/odom` for runaway speed ($v > v_{\max}$) using reason `'speed_limit'`. Its subscription is disabled in the deployed configuration (not needed under current operational conditions) but the handler remains wired for future use.

---

## 7. Stopping Distance Bound

The worst-case scenario is an obstacle appearing at exactly $d_{\min} + \epsilon$ at the moment a confirmation streak resets (e.g., the previous scan was a non-detection due to occlusion). The robot then travels for $\tau_\text{trig}$ before the stop triggers:

$$
d_\text{approach} = v \cdot \tau_\text{trig} = v \cdot \frac{n_\text{trig}}{f_s}
$$

After the stop signal goes high, the motion driver sends zero velocity and the robot decelerates over braking distance $d_\text{brake}$ (mechanical, depends on floor friction and speed). For a collision-free stop:

$$
d_{\min} > d_\text{approach} + d_\text{brake} = v \cdot \frac{n_\text{trig}}{f_s} + d_\text{brake}
$$

Empirically, 0.30 m satisfies this inequality across all tested configurations.

**Note on scan rate dependence.** The debounce duration scales inversely with $f_s$. If the MS200 were run at a higher scan rate or `n_trig` were reduced, the trigger latency would shrink, allowing a smaller $d_{\min}$ without sacrificing safety — or equivalently, allowing higher cruise speeds. The current 5-scan / 15 Hz setting was tuned on the physical platform.

---

## 8. Parameters

| Parameter | Deployed Value | Meaning |
|---|---|---|
| `min_obstacle_distance_m` | 0.30 m | Proximity threshold $d_{\min}$ |
| `max_linear_speed_mps` | 1.0 m/s | Speed gate threshold (odometry trigger, currently inactive) |
| `mask_angles_deg` | [120.0, 240.0] | Blind-spot sector excluded from proximity check |
| `trigger_count` | 5 | Consecutive obstacle detections required to trigger stop |
| `clear_count` | 5 | Consecutive clear scans required to release stop |

Parameters `trigger_count` and `clear_count` are not overridden in `safety_params.yaml` and use code defaults.

---

## 9. Known Limitations

**Blind-spot coverage.** The rear 120° sector is fully excluded. An obstacle appearing directly behind the robot while it reverses (if the planner ever issues reverse commands) would not be detected. This is currently a non-issue but must be revisited if reverse-motion trajectories are introduced.

**Single-plane LiDAR.** The MS200 is a 2D scanner in the horizontal plane at a fixed mounting height. Obstacles lower than the scan plane (floor debris, low-profile objects) or higher than the scan plane (overhanging structures) are invisible to the AEB. The 0.30 m threshold was tuned for cardboard-box obstacles in the test arena; it provides no formal guarantee against atypical obstacle geometries.

**Streak-reset vulnerability.** The N-consecutive filter resets the counter on a single non-detection. A flickering reading (obstacle alternately detected and dropped due to specular reflection) can indefinitely postpone triggering if the detection rate is exactly 1-on/1-off. In practice this was not observed with cardboard obstacles, but glass or metallic surfaces could produce this pathology.

**No directional awareness.** The proximity check takes the global minimum over all non-masked valid readings. The node does not know whether the obstacle is in the direction of intended motion. A stationary wall to the side of a turn could trigger a stop even if the planned path clears it. Directional masking (filtering to the forward $\pm\alpha$ cone when moving forward) would reduce spurious triggers at the cost of additional coupling with the motion state.
