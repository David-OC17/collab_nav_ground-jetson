"""
arena_marker_localizer.service_node
─────────────────────────────────────────────────────────────────────────────
ROS 2 service node. One service:

    /localize_markers   (arena_marker_localizer/srv/LocalizeMarkers)

Request:
    string video_path
    string optitrack_csv

Response:
    bool                                  success
    string                                message
    arena_marker_localizer/MarkerPose[]   markers

All other parameters are exposed as rclpy parameters (see _declare_parameters
below). This is a synchronous service — the request blocks until the
pipeline completes; there is no per-stage feedback (you chose service
over action for this one).
"""

from __future__ import annotations

import os
from typing import List, Optional

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geometry_msgs.msg import Pose, Point, Quaternion, Pose2D

from arena_marker_localizer_interfaces.srv import LocalizeMarkers
from arena_marker_localizer_interfaces.msg import MarkerPose

from arena_marker_localizer.pipeline import (
    PipelineConfig, run_pipeline, MarkerResult,
)
from arena_marker_localizer.transforms import (
    StaticTransform6DoF, OptiTrackAxisConfig, R_to_quaternion, euler_zyx_to_R,
)
from arena_marker_localizer.marker_detection import DictionaryConfig
from arena_marker_localizer.quality import QualityConfig
from arena_marker_localizer.aggregation import AggregationConfig
from arena_marker_localizer.debug_image import (
    DebugImageConfig, render_debug_image,
)


class MarkerLocalizerService(Node):

    def __init__(self):
        super().__init__("marker_localizer_service")

        self._declare_parameters()

        self._srv = self.create_service(
            LocalizeMarkers,
            "localize_markers",
            self.handle_request,
        )
        self.get_logger().info(
            "Service /localize_markers is ready (arena_marker_localizer)."
        )

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    def _declare_parameters(self):
        # ── Intrinsics ─────────────────────────────────────────────────
        self.declare_parameter("intrinsics_path", "")
        self.declare_parameter("intrinsics.camera_matrix_key", "camera_matrix")
        self.declare_parameter("intrinsics.dist_key",          "dist_coeff")

        # ── ArUco dictionaries — list of strings "DICT_NAME:size_m" ───
        # Example: ["DICT_4X4_50:0.10", "DICT_5X5_100:0.08"]
        self.declare_parameter(
            "dictionaries",
            ["DICT_4X4_50:0.10"],
        )

        # ── Quality ────────────────────────────────────────────────────
        self.declare_parameter("quality.blur_thresh",     60.0)
        self.declare_parameter("quality.artifact_thresh",  2.0)

        # ── Detection gates ────────────────────────────────────────────
        self.declare_parameter("max_reproj_err_px", 4.0)

        # ── Static transform: drone-from-cam (6 params) ────────────────
        for name in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.declare_parameter(f"T_drone_from_cam.{name}", 0.0)

        # ── Static transform: map-from-opti (6 params) ─────────────────
        for name in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.declare_parameter(f"T_map_from_opti.{name}", 0.0)

        # ── OptiTrack axis config ──────────────────────────────────────
        self.declare_parameter("optitrack.yaw_axis", "z")    # "z" or "y"
        self.declare_parameter("optitrack.x_dir",    1)      # +1 or -1
        self.declare_parameter("optitrack.y_dir",    1)

        # ── Aggregation ────────────────────────────────────────────────
        self.declare_parameter("aggregation.mad_k",            3.5)
        self.declare_parameter("aggregation.min_observations", 2)
        self.declare_parameter("aggregation.max_iterations",   100)
        self.declare_parameter("aggregation.convergence_eps",  1e-5)
        self.declare_parameter("max_obs_per_marker",           200)

        # ── Arena / grid sizing ────────────────────────────────────────
        self.declare_parameter("arena.width_m",               3.14)
        self.declare_parameter("arena.height_m",              3.14)
        self.declare_parameter("grid.resolution_m_per_cell",  0.05)

        # ── Debug image ────────────────────────────────────────────────
        self.declare_parameter("debug.output_path",
                               "/tmp/marker_localizer_debug.png")
        self.declare_parameter("debug.px_per_cell", 12)

        self.declare_parameter("verbose", False)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get(self, name, default=None):
        return self.get_parameter(name).value if self.has_parameter(name) \
            else default

    def _parse_dictionaries(self) -> List[DictionaryConfig]:
        raw = self._get("dictionaries") or []
        out: List[DictionaryConfig] = []
        for entry in raw:
            try:
                name, size = entry.split(":")
                out.append(DictionaryConfig(
                    name=name.strip(), marker_size_m=float(size.strip()),
                ))
            except Exception:
                self.get_logger().warn(
                    f"Ignoring malformed dictionaries entry: {entry!r}. "
                    f"Expected 'DICT_NAME:size_m'."
                )
        if not out:
            self.get_logger().warn(
                "No valid dictionaries configured; falling back to "
                "DICT_4X4_50 @ 0.10 m."
            )
            out.append(DictionaryConfig(name="DICT_4X4_50",
                                        marker_size_m=0.10))
        return out

    def _build_static(self, prefix: str) -> StaticTransform6DoF:
        return StaticTransform6DoF(
            x=float(self._get(f"{prefix}.x")),
            y=float(self._get(f"{prefix}.y")),
            z=float(self._get(f"{prefix}.z")),
            roll=float(self._get(f"{prefix}.roll")),
            pitch=float(self._get(f"{prefix}.pitch")),
            yaw=float(self._get(f"{prefix}.yaw")),
        )

    def _build_pipeline_cfg(self) -> PipelineConfig:
        return PipelineConfig(
            intrinsics_path                = str(self._get("intrinsics_path")),
            intrinsics_camera_matrix_key   = str(self._get("intrinsics.camera_matrix_key")),
            intrinsics_dist_key            = str(self._get("intrinsics.dist_key")),
            dictionaries     = self._parse_dictionaries(),
            quality          = QualityConfig(
                blur_thresh     = float(self._get("quality.blur_thresh")),
                artifact_thresh = float(self._get("quality.artifact_thresh")),
            ),
            T_drone_from_cam = self._build_static("T_drone_from_cam"),
            T_map_from_opti  = self._build_static("T_map_from_opti"),
            optitrack_axis   = OptiTrackAxisConfig(
                yaw_axis = str(self._get("optitrack.yaw_axis")),
                x_dir    = int(self._get("optitrack.x_dir")),
                y_dir    = int(self._get("optitrack.y_dir")),
            ),
            max_reproj_err_px = float(self._get("max_reproj_err_px")),
            aggregation = AggregationConfig(
                mad_k            = float(self._get("aggregation.mad_k")),
                min_observations = int(self._get("aggregation.min_observations")),
                max_iterations   = int(self._get("aggregation.max_iterations")),
                convergence_eps  = float(self._get("aggregation.convergence_eps")),
            ),
            max_obs_per_marker = int(self._get("max_obs_per_marker")),
            resolution_m_per_cell = float(self._get("grid.resolution_m_per_cell")),
            grid_width_cells  = math.ceil(
                float(self._get("arena.width_m"))  / float(self._get("grid.resolution_m_per_cell"))
            ),
            grid_height_cells = math.ceil(
                float(self._get("arena.height_m")) / float(self._get("grid.resolution_m_per_cell"))
            ),
            verbose = bool(self._get("verbose")),
        )

    @staticmethod
    def _marker_result_to_msg(r: MarkerResult) -> MarkerPose:
        msg = MarkerPose()
        msg.id = int(r.marker_id)

        # ── pose_3d ──
        # Build a 3D rotation matrix with roll=0, pitch=0, yaw=r.yaw_rad,
        # then convert to quaternion. Roll/pitch information is *lost*
        # by the aggregation step (which only retains yaw), so we model
        # the marker as flat-on-floor for the 3D output too.
        R = euler_zyx_to_R(0.0, 0.0, r.yaw_rad)
        qx, qy, qz, qw = R_to_quaternion(R)
        msg.pose_3d = Pose(
            position=Point(
                x=float(r.position_m[0]),
                y=float(r.position_m[1]),
                z=float(r.position_m[2]),
            ),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

        # ── pose_2d ──
        msg.pose_2d = Pose2D(
            x=float(r.position_m[0]),
            y=float(r.position_m[1]),
            theta=float(r.yaw_rad),
        )

        msg.cell_x = int(r.cell_x)
        msg.cell_y = int(r.cell_y)
        msg.n_observations = int(r.n_observations)
        return msg

    # -----------------------------------------------------------------------
    # Service callback
    # -----------------------------------------------------------------------

    def handle_request(self, request, response):
        video = request.video_path
        csv   = request.optitrack_csv

        if not video or not os.path.isfile(video):
            response.success = False
            response.message = f"video_path does not exist: {video!r}"
            return response
        if not csv or not os.path.isfile(csv):
            response.success = False
            response.message = f"optitrack_csv does not exist: {csv!r}"
            return response

        try:
            cfg = self._build_pipeline_cfg()
        except Exception as e:
            response.success = False
            response.message = f"config build failed: {e}"
            return response

        if not cfg.intrinsics_path:
            response.success = False
            response.message = ("intrinsics_path parameter is empty. "
                                "Set it via `ros2 param set ...`.")
            return response

        try:
            self.get_logger().info(
                f"Localizing markers: video={video!r}, csv={csv!r}"
            )
            results, drone_poses = run_pipeline(video, csv, cfg)
        except Exception as e:
            self.get_logger().error(f"Pipeline crashed: {e}")
            response.success = False
            response.message = f"pipeline exception: {e}"
            return response

        msgs: List[MarkerPose] = [
            self._marker_result_to_msg(r)
            for r in sorted(results.values(), key=lambda x: x.marker_id)
        ]

        # ── Debug reference image ────────────────────────────────────
        # Best-effort: write the debug PNG but don't fail the service
        # if it can't be written (disk full, permission, etc.).
        try:
            flip = np.diag([float(cfg.optitrack_axis.x_dir),
                            float(cfg.optitrack_axis.y_dir),
                            1.0, 1.0])
            T_opti_to_map = cfg.T_map_from_opti.as_matrix() @ flip
            dbg_cfg = DebugImageConfig(
                grid_width_cells      = cfg.grid_width_cells,
                grid_height_cells     = cfg.grid_height_cells,
                resolution_m_per_cell = cfg.resolution_m_per_cell,
                px_per_cell           = int(self._get("debug.px_per_cell")),
                T_opti_to_map         = T_opti_to_map,
            )
            out_path = str(self._get("debug.output_path"))
            render_debug_image(results.values(), drone_poses, dbg_cfg, out_path)
            self.get_logger().info(f"Debug reference image: {out_path}")
        except Exception as e:
            self.get_logger().warn(f"Debug image render failed: {e}")

        response.success = True
        response.message = f"Localized {len(msgs)} marker(s)."
        response.markers = msgs
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MarkerLocalizerService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()