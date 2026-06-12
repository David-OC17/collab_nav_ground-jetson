# Knowledge Base — Report Reference Index

Graduation project: cooperative UAV–AMR navigation in a GPS-denied arena.  
**LaTeX class:** `\documentclass[conference]{IEEEtran}`  
**Authors:** Ortiz · Romo · Rosales · Pulido · Gonzales  
**Institution:** Tecnológico de Monterrey, in partnership with OMRON Automation and Nuclea Solutions.

---

## Source Documentation

All subsystem theory, equations, and implementation detail lives in the DOCS.md files below.  
The report text is derived from these sources; do not duplicate prose here.

| Subsystem | Documentation |
|---|---|
| Top-level system overview, hardware, node graph | `collab_nav_ground-jetson/DOCS.md` |
| UAV PID controller, spline trajectory, fault detector | `collab_nav_uav/src/tello_pos_control/DOCS.md` |
| Aerial image stitching, SIFT/SuperPoint, RF quality gate | `arena_map_builder/DOCS.md` |
| ArUco PnP localization, transform chain, bias calibration | `arena_marker_localizer/DOCS.md` |
| OptiTrack → velocity bridge (VRPN, EMA) | `amr_optitrack/DOCS.md` |
| Emergency braking (AEB), streak detector | `emergency_stop/DOCS.md` |
| Map fusion (elementwise priority rule) | `fusion/DOCS.md` |
| Mission state machine (10-stage FSM) | `mission_orchestrator/DOCS.md` |
| Bayesian occupancy mapping, log-odds, Bresenham | `world_mapper/DOCS.md` |
| A* planner, cubic spline, feedback linearization | `trajectory_planner/DOCS.md` |
| LiDAR driver (OraDAR MS200 wire protocol) | `oradar_ros/DOCS.md` |
| AMR hardware layer: driver, wheel odometry, controller | `collab_nav_ground-rasp/src/amr_bringup/DOCS.md` |
| EKF (5-state unicycle, wheel+IMU+VIO, adaptive gate) | `collab_nav_ground-rasp/src/ekf_amr/DOCS.md` |
| Security middleware (PKI, RSA-PSS, AES-256-GCM) | `security_middleware/DOCS.md` |

---

## Professor's Required Points → Report Coverage

| Requirement (pts) | Covered by |
|---|---|
| GPS-denied navigation (required) | EKF + fusion + planner sections |
| UAV aerial exploration (required) | tello_pos_control + arena_map_builder sections |
| Manual Kalman filter (required) | EKF section |
| Cooperative localization (required) | arena_marker_localizer + mission FSM sections |
| Decentralized software — ROS 2 (required) | System architecture section |
| Deep learning for vision (required) | SuperPoint/ONNX subsection in arena_map_builder |
| Physical safety / AEB (required) | emergency_stop section |
| Cybersecurity (required) | security_middleware section |
| Dynamic speed adjustment (extra) | trajectory_planner §trapezoidal profile |
| Embedded DL inference (extra) | SuperPoint ONNX on Jetson GPU |
| Quantitative improvement metrics (extra) | Results section (experimental data) |

---

## Theoretical Background Topics

Each of these needs a brief, self-contained treatment in the Background section:

- Unicycle kinematics and feedback linearization
- Extended Kalman filter (linearization, Jacobian, Joseph form)
- Bayesian occupancy grids and log-odds updates
- SIFT feature detection and Lowe's ratio test
- Image stitching: homography vs. similarity transform, RANSAC
- Laplacian pyramid blending
- ArUco detection and PnP pose estimation (IPPE_SQUARE)
- A* search with octile heuristic; arc-length cubic spline
- RSA-PSS signatures and AES-256-GCM encryption basics

---

## Open Questions (resolve before drafting)

- Which subsystem receives the full §5 treatment (requirements, traceability, implementation, tests)?
- Is experimental data available (navigation success rate, EKF RMSE, stitching coverage), or placeholders?
- Language: English or Spanish?
- Approximate target page count for the IEEEtran conference format?
- Does `report/report.text` contain a prior draft to build on?
