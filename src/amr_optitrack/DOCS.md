# `amr_optitrack` — OptiTrack Pose Bridge

## 1. Role in the System

The OptiTrack motion-capture system tracks rigid bodies in the arena at ~120 Hz and publishes their 6-DoF poses via the `optitrack_client` VRPN bridge. Those poses carry position and orientation but **no velocity** — velocity must be derived.

`amr_optitrack` is the adaptation layer between the raw OptiTrack stream and the system's internal odometry representation. It:

1. Filters the VRPN stream to the AMR rigid body.
2. Estimates body-frame linear and angular velocity by finite difference between consecutive poses.
3. Applies an exponential moving-average (EMA) filter to suppress velocity noise from pose quantisation.
4. Publishes the combined pose + velocity as a standard ROS `Odometry` message consumed by the EKF (`ekf_amr`).

This node is the source of ground-truth absolute pose for the AMR. When it is available (OptiTrack is running and the AMR body is tracked), it anchors the EKF against long-term odometric drift.

---

## 2. Problem Statement

Let the AMR pose at time $t_k$ be $\mathbf{p}_k = (x_k,\, y_k,\, \theta_k)^\top$ in the world frame, delivered as a `PoseStamped` from the VRPN bridge. The raw message carries **no velocity information**; the EKF requires $(\dot{x},\, \dot{y},\, \dot{\theta})$ or equivalently body-frame $(v_x,\, v_y,\, \omega_z)$ to form its measurement update.

The two sub-problems are:

- **Velocity estimation:** Compute a velocity estimate from consecutive noisy pose samples with non-uniform timestamps.
- **Noise rejection:** The OptiTrack pose is subject to quantisation at the VRPN network layer and occasional marker occlusion, both of which inject transient outliers into a naive finite-difference estimate.

---

## 3. Velocity Estimation Pipeline

### 3.1 Finite Difference

Given two consecutive valid poses at times $t_{k-1}$ and $t_k$, with $\Delta t = t_k - t_{k-1}$, the world-frame velocity is approximated by a first-order backward difference:

$$
\dot{x}^w_k = \frac{x_k - x_{k-1}}{\Delta t}, \qquad
\dot{y}^w_k = \frac{y_k - y_{k-1}}{\Delta t}, \qquad
\dot{\theta}_k = \frac{\text{wrap}(\theta_k - \theta_{k-1})}{\Delta t}
$$

where $\text{wrap}(\cdot)$ maps angular differences to $(-\pi, \pi]$:

$$\text{wrap}(\alpha) = \text{atan2}(\sin\alpha,\, \cos\alpha)$$

This eliminates aliasing when the heading crosses the $\pm\pi$ discontinuity.

### 3.2 $\Delta t$ Guard Band

The finite difference amplifies noise as $\Delta t \to 0$. Two degenerate cases are explicitly rejected:

| Condition | Reason | Action |
|---|---|---|
| $\Delta t \leq 5$ ms | Duplicate or nearly-simultaneous samples — division would amplify quantisation noise by $\times 200$ | Skip; reset bookkeeping |
| $\Delta t > 500$ ms | Stale pair after topic gap or tracker loss — velocity estimate is meaningless | Skip; reset bookkeeping |

The effective operating range is $\Delta t \in (5, 500]$ ms. At 120 Hz the nominal $\Delta t \approx 8.3$ ms sits comfortably within this band.

### 3.3 World → Body Frame Rotation

The EKF uses body-frame velocities (forward/lateral/yaw) rather than world-frame. The rotation from world to body is:

$$
\begin{pmatrix} v_x \\ v_y \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos\theta_k & \sin\theta_k \\ -\sin\theta_k & \cos\theta_k \end{pmatrix}}_{R(\theta_k)^\top}
\begin{pmatrix} \dot{x}^w_k \\ \dot{y}^w_k \end{pmatrix}
$$

This is the standard body←world rotation (transposed rotation matrix), valid for a planar rigid body whose heading is $\theta_k$. Angular rate $\omega_z = \dot{\theta}_k$ is frame-invariant for planar rotation and requires no transformation.

### 3.4 EMA Low-Pass Filter

Marker-induced position jitter and sub-millisecond VRPN latency variations produce high-frequency noise in the raw finite-difference velocity. An exponential moving average (EMA) with smoothing factor $\alpha$ is applied to each velocity channel independently:

$$
\hat{v}_{x,k} = \alpha \cdot v_{x,k}^{\text{raw}} + (1-\alpha)\cdot \hat{v}_{x,k-1}
$$

and equivalently for $\hat{v}_y$ and $\hat{\omega}_z$. The effective time constant of the EMA at sample rate $f_s$ is:

$$
\tau_\text{EMA} = \frac{\Delta t}{\alpha} = \frac{1}{\alpha \cdot f_s}
$$

At the deployed value $\alpha = 0.1$ and $f_s = 120$ Hz:

$$
\tau_\text{EMA} = \frac{1}{0.1 \times 120} \approx 83\text{ ms}
$$

This cuts high-frequency velocity noise well above the AMR's mechanical bandwidth (~1–2 Hz) while introducing negligible lag on the slow-moving chassis.

### 3.5 Pseudocode

```
Procedure OptiTrackVelocityUpdate(msg):
    if msg.frame_id ≠ 'AMR': return

    now ← msg.stamp.sec + msg.stamp.nanosec × 1e-9
    if no prior sample:
        store(msg, now); return

    Δt ← now − t_prev
    if Δt ≤ 0.005 or Δt > 0.5:
        store(msg, now); return             # guard band

    # Finite difference in world frame
    Δx ← msg.x − prev.x
    Δy ← msg.y − prev.y
    Δθ ← wrap(yaw(msg.q) − yaw(prev.q))

    dx_w ← Δx / Δt
    dy_w ← Δy / Δt
    dθ   ← Δθ / Δt

    # World → body frame
    c ← cos(yaw(msg.q));  s ← sin(yaw(msg.q))
    vx_raw ←  c·dx_w + s·dy_w
    vy_raw ← −s·dx_w + c·dy_w

    # EMA filter (α = 0.1)
    v̂x ← α·vx_raw + (1−α)·v̂x
    v̂y ← α·vy_raw + (1−α)·v̂y
    ω̂z ← α·dθ    + (1−α)·ω̂z

    publish Odometry(pose=msg.pose, twist=(v̂x, v̂y, ω̂z))
    store(msg, now)
```

---

## 4. Message Interface

```
Subscriptions:
  /optitrack/rigid_body  (geometry_msgs/PoseStamped, ~120 Hz, BEST_EFFORT QoS)
      header.frame_id filtered: only 'AMR' frames processed

Publications:
  /amr/pose  (nav_msgs/Odometry, ~120 Hz)
      header.frame_id  = 'odom'
      child_frame_id   = 'base_link'
      pose.pose        = position + orientation from OptiTrack
      twist.twist      = EMA-smoothed body-frame velocity
```

The published `Odometry` message uses `frame_id = 'odom'` and `child_frame_id = 'base_link'` to match the ROS convention expected by the EKF. The pose covariance and twist covariance fields are left at their default (zero), which signals to the EKF that it should use its own configured measurement noise matrices.

---

## 5. Parameters

| Parameter | Default | Description |
|---|---|---|
| `vel_alpha` | 0.1 | EMA smoothing factor $\alpha \in (0,1]$; larger → less smoothing, faster response |

---

## 6. Assumptions and Limitations

**Single rigid body.** The node filters on `frame_id == 'AMR'`. If OptiTrack is tracking multiple bodies and the VRPN bridge mixes their streams onto the same topic, only messages labelled `'AMR'` are processed; others are silently discarded.

**No covariance propagation.** The Odometry message carries no velocity covariance. Downstream consumers (EKF) must supply their own measurement noise model. The EMA time constant of ~83 ms is not reflected in the published uncertainty.

**Velocity lag.** The EMA introduces a phase delay of approximately $\tau_\text{EMA} \approx 83$ ms at the $-3$ dB point. For the EKF this is acceptable because position is directly observed (no integration); velocity is a derived secondary quantity. During fast manoeuvres the velocity estimate will lag the true body velocity.

**No outlier gating on pose.** Sudden large pose jumps (e.g., from a momentary marker swap in OptiTrack) pass through the $\Delta t$ guard unchanged. The EMA dampens their effect on velocity but the pose itself is published immediately. The EKF's Mahalanobis distance gate provides the downstream outlier rejection.
