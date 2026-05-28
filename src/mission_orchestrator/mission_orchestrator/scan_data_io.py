"""
Serialization helpers for recorded scan data.

Saves/loads aruco poses (PoseWithCovarianceStamped) and the drone map
(OccupancyGrid) to/from YAML files under recorded_data/scanX/.

Directory conventions (relative to workspace root):
  drone scan input  : src/arena_map_builder/data/drone_scans/scanX/
  recorded outputs  : src/mission_orchestrator/recorded_data/scanX/

File names inside recorded_data/scanX/:
  aruco_amr_pose.yaml   — AMR marker PoseWithCovarianceStamped
  aruco_goal_pose.yaml  — goal marker PoseWithCovarianceStamped
  drone_map.yaml        — OccupancyGrid header + info + base64-encoded data
"""
from __future__ import annotations

import base64
import os
import struct

import yaml

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid


# ── Path helpers ─────────────────────────────────────────────────────────────

def recorded_data_dir(workspace_root: str, scan_id: int) -> str:
    return os.path.join(workspace_root, 'src', 'mission_orchestrator',
                        'recorded_data', f'scan{scan_id}')


def scan_video_dir(workspace_root: str, scan_id: int) -> str:
    return os.path.join(workspace_root, 'src', 'arena_map_builder',
                        'data', 'drone_scans', f'scan{scan_id}')


# ── PoseWithCovarianceStamped ─────────────────────────────────────────────────

def save_pose(path: str, msg: PoseWithCovarianceStamped) -> None:
    p = msg.pose.pose
    d = {
        'header': {'frame_id': msg.header.frame_id},
        'pose': {
            'position': {
                'x': float(p.position.x),
                'y': float(p.position.y),
                'z': float(p.position.z),
            },
            'orientation': {
                'x': float(p.orientation.x),
                'y': float(p.orientation.y),
                'z': float(p.orientation.z),
                'w': float(p.orientation.w),
            },
        },
        'covariance': [float(v) for v in msg.pose.covariance],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        yaml.dump(d, fh, default_flow_style=False)


def load_pose(path: str) -> PoseWithCovarianceStamped:
    with open(path, 'r') as fh:
        d = yaml.safe_load(fh)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = d['header']['frame_id']
    p = d['pose']
    pos = p['position']
    ori = p['orientation']
    msg.pose.pose.position.x = float(pos['x'])
    msg.pose.pose.position.y = float(pos['y'])
    msg.pose.pose.position.z = float(pos['z'])
    msg.pose.pose.orientation.x = float(ori['x'])
    msg.pose.pose.orientation.y = float(ori['y'])
    msg.pose.pose.orientation.z = float(ori['z'])
    msg.pose.pose.orientation.w = float(ori['w'])
    for i, v in enumerate(d['covariance']):
        msg.pose.covariance[i] = float(v)
    return msg


# ── OccupancyGrid ─────────────────────────────────────────────────────────────

def save_grid(path: str, msg: OccupancyGrid) -> None:
    # Encode int8 data as base64 for compact single-file storage.
    n = len(msg.data)
    raw = struct.pack(f'{n}b', *msg.data)
    data_b64 = base64.b64encode(raw).decode('ascii')

    op = msg.info.origin.position
    oo = msg.info.origin.orientation
    d = {
        'header': {'frame_id': msg.header.frame_id},
        'info': {
            'width': int(msg.info.width),
            'height': int(msg.info.height),
            'resolution': float(msg.info.resolution),
            'origin': {
                'position': {
                    'x': float(op.x), 'y': float(op.y), 'z': float(op.z),
                },
                'orientation': {
                    'x': float(oo.x), 'y': float(oo.y),
                    'z': float(oo.z), 'w': float(oo.w),
                },
            },
        },
        'data_b64': data_b64,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        yaml.dump(d, fh, default_flow_style=False)


def load_grid(path: str) -> OccupancyGrid:
    with open(path, 'r') as fh:
        d = yaml.safe_load(fh)
    msg = OccupancyGrid()
    msg.header.frame_id = d['header']['frame_id']
    info = d['info']
    msg.info.width = int(info['width'])
    msg.info.height = int(info['height'])
    msg.info.resolution = float(info['resolution'])
    op = info['origin']['position']
    oo = info['origin']['orientation']
    msg.info.origin.position.x = float(op['x'])
    msg.info.origin.position.y = float(op['y'])
    msg.info.origin.position.z = float(op['z'])
    msg.info.origin.orientation.x = float(oo['x'])
    msg.info.origin.orientation.y = float(oo['y'])
    msg.info.origin.orientation.z = float(oo['z'])
    msg.info.origin.orientation.w = float(oo['w'])
    raw = base64.b64decode(d['data_b64'])
    msg.data = list(struct.unpack(f'{len(raw)}b', raw))
    return msg


# ── Convenience: validate saved data exists ───────────────────────────────────

SAVED_FILES = ('aruco_amr_pose.yaml', 'aruco_goal_pose.yaml', 'drone_map.yaml')


def assert_saved_data_exists(data_dir: str, scan_id: int) -> None:
    missing = [f for f in SAVED_FILES
               if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Saved scan data missing in {data_dir!r}: {missing}\n"
            f"Run:  python3 save_scan_data.py --scan-id {scan_id}")
