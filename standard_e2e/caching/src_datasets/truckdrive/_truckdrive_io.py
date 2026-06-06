"""On-disk access helpers for TruckDrive (extracted scenes).

TruckDrive ships per scene as one zip per modality (``camera.zip``,
``lidar.zip``, ``annotations.zip``, ``calibrations.zip``, ``poses.zip``, ...).
**Extract them first** (``scripts/extract_truckdrive.sh``), expanding each zip
into a folder of the same name so the on-disk tree matches the official devkit
layout::

    <root>/scene_28_1/
        calibrations/calib_tf_tree_full.json
        calibrations/calib_camera_leopard_<camera>.json
        camera/leopard/<camera>/images/<sync>_<ts>.jpg
        lidar/aeva/joint_lidars/points/<sync>_<ts>.bin
        annotations/bounding_boxes/<sync>_<ts>.json
        poses/gt_trajectory.txt

Filenames are ``<sync_id>_<normalized_timestamp_ns>.<ext>``. The ``sync_id`` (the
4-digit prefix) is the cross-sensor synchronization key: sensors that fired for
the same synchronized snapshot share a ``sync_id`` even when their exact
timestamps differ slightly, so all matching across modalities is keyed on it
(matching the devkit). This module discovers scene directories, reads the
per-scene calibration / pose / sensor files, and exposes ``sync_id -> path``
maps for the per-frame lookups the processor needs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Relative layout (post-extraction; matches the devkit ``dataset_details.py``).
_TF_TREE_RELPATH = "calibrations/calib_tf_tree_full.json"
_CALIB_DIR = "calibrations"
_CAMERA_ROOT = "camera/leopard"
_AEVA_POINTS_DIR = "lidar/aeva/joint_lidars/points"
_BOXES_DIR = "annotations/bounding_boxes"
_TRAJECTORY_RELPATH = "poses/gt_trajectory.txt"

# Aeva joint cloud: float64, 11 cols
# (x, y, z, intensity, velocity, reflectivity, time_offset_ns, sensor_id, vx, vy, vz).
_AEVA_NUM_COLS = 11

_SYNC_STEM_RE = re.compile(r"^(?P<sync>\d+)_(?P<ts>\d+)$", re.ASCII)
_CAMERA_CALIB_RE = re.compile(r"^calib_camera_leopard_(?P<name>.+)\.json$", re.ASCII)


@dataclass(frozen=True)
class SceneRef:
    """Picklable locator for one extracted TruckDrive scene.

    Attributes:
        path: the scene directory (holds ``calibrations/``, ``camera/``, ...).
        scene_id: the scene directory name (e.g. ``scene_28_1``), used as the
            ``StandardFrameData.segment_id``.
    """

    path: str
    scene_id: str

    @property
    def trajectory_path(self) -> str:
        return os.path.join(self.path, _TRAJECTORY_RELPATH)


def discover_scenes(input_path: str) -> list[SceneRef]:
    """Find every extracted TruckDrive scene under ``input_path``.

    A scene is any directory that holds ``poses/gt_trajectory.txt`` (and hence
    has been extracted). Raises ``FileNotFoundError`` -- pointing at the
    required extraction step -- when none are found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"TruckDrive input path not found: {root}")

    scenes: list[SceneRef] = []
    # The trajectory file is the cheapest unique marker of an extracted scene.
    for marker in root.rglob(_TRAJECTORY_RELPATH):
        scene_dir = marker.parent.parent
        scenes.append(SceneRef(str(scene_dir), scene_dir.name))
    if not scenes:
        raise FileNotFoundError(
            f"No extracted TruckDrive scenes under {root} (expected "
            f"<scene>/{_TRAJECTORY_RELPATH}). Extract the per-scene modality "
            f"zips first, e.g. scripts/extract_truckdrive.sh."
        )
    return sorted(scenes, key=lambda r: r.scene_id)


def parse_sync_stem(filename: str) -> tuple[int, int] | None:
    """Parse a ``<sync_id>_<timestamp_ns>`` stem -> ``(sync_id, timestamp_ns)``.

    Returns ``None`` for names that do not match (so callers can skip them).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = _SYNC_STEM_RE.match(stem)
    if match is None:
        return None
    return int(match.group("sync")), int(match.group("ts"))


def _synced_files(folder: str, ext: str) -> dict[int, str]:
    """Map ``sync_id -> path`` for ``*.<ext>`` files in ``folder`` (``{}`` if
    the folder is absent). On the rare duplicate ``sync_id`` the nearest is not
    resolved here; the last lexicographic entry wins."""
    if not os.path.isdir(folder):
        return {}
    out: dict[int, str] = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(f".{ext}"):
            continue
        parsed = parse_sync_stem(name)
        if parsed is None:
            continue
        out[parsed[0]] = os.path.join(folder, name)
    return out


def camera_names(scene: SceneRef) -> list[str]:
    """Physical camera folder names present under ``camera/leopard`` (sorted)."""
    cam_root = os.path.join(scene.path, _CAMERA_ROOT)
    if not os.path.isdir(cam_root):
        return []
    return sorted(
        name
        for name in os.listdir(cam_root)
        if os.path.isdir(os.path.join(cam_root, name, "images"))
    )


def camera_sync_paths(scene: SceneRef, camera_name: str) -> dict[int, str]:
    """``sync_id -> image path`` for one camera."""
    return _synced_files(
        os.path.join(scene.path, _CAMERA_ROOT, camera_name, "images"), "jpg"
    )


def aeva_sync_paths(scene: SceneRef) -> dict[int, str]:
    """``sync_id -> Aeva joint point-cloud path``."""
    return _synced_files(os.path.join(scene.path, _AEVA_POINTS_DIR), "bin")


def box_sync_paths(scene: SceneRef) -> dict[int, str]:
    """``sync_id -> 3D bounding-box JSON path``."""
    return _synced_files(os.path.join(scene.path, _BOXES_DIR), "json")


def read_trajectory(scene: SceneRef) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Read ``poses/gt_trajectory.txt``.

    Rows are ``SYNC_KEY TIMESTAMP X Y Z R_X R_Y R_Z R_W`` (scalar-last
    quaternion), in a per-scene local world frame. Returns ``(sync_ids,
    timestamps_s, quats_and_positions)`` where ``timestamps_s`` is ``(N,)``
    seconds and ``poses_raw`` is ``(N, 7)`` ``[x, y, z, qx, qy, qz, qw]`` left
    for the geometry helper to assemble into 4x4 transforms.
    """
    data = np.loadtxt(scene.trajectory_path, skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] < 9:
        raise ValueError(
            f"{scene.trajectory_path}: expected >=9 columns "
            f"(SYNC_KEY TIMESTAMP X Y Z R_X R_Y R_Z R_W); got {data.shape[1]}"
        )
    sync_ids = [int(round(v)) for v in data[:, 0]]
    timestamps_s = data[:, 1].astype(np.float64)
    poses_raw = data[:, 2:9].astype(np.float64)  # x, y, z, qx, qy, qz, qw
    return sync_ids, timestamps_s, poses_raw


def read_pose_sync_ids(scene: SceneRef) -> list[int]:
    """Read just the ``SYNC_KEY`` column of the trajectory (cheap; for the
    converter's frame enumeration)."""
    col = np.loadtxt(scene.trajectory_path, skiprows=1, usecols=(0,))
    return [int(round(v)) for v in np.atleast_1d(col)]


def read_calibrations(
    scene: SceneRef,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the TF tree and per-camera intrinsics for a scene.

    Returns ``(tf_tree, camera_calibs)`` where ``camera_calibs[name]`` has keys
    ``intrinsics`` (3x3 float32 ``K``), ``width`` and ``height``. Camera calib
    JSONs follow the ROS ``CameraInfo`` schema; the public images are already
    rectified/undistorted (``D == 0``), so no distortion is carried.
    """
    with open(os.path.join(scene.path, _TF_TREE_RELPATH), encoding="utf-8") as fh:
        tf_tree = json.load(fh)

    calib_dir = os.path.join(scene.path, _CALIB_DIR)
    camera_calibs: dict[str, dict[str, Any]] = {}
    for name in sorted(os.listdir(calib_dir)):
        match = _CAMERA_CALIB_RE.match(name)
        if match is None:
            continue
        with open(os.path.join(calib_dir, name), encoding="utf-8") as fh:
            calib = json.load(fh)
        camera_calibs[match.group("name")] = {
            "intrinsics": np.asarray(calib["K"], dtype=np.float32).reshape(3, 3),
            "width": int(calib["width"]),
            "height": int(calib["height"]),
        }
    return tf_tree, camera_calibs


def read_image(path: str) -> np.ndarray:
    """Read a camera JPEG as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def read_aeva_xyz(path: str) -> np.ndarray:
    """Read the Aeva joint cloud and return its ``(N, 3)`` float32 xyz columns.

    The binary is ``float64`` with 11 columns per point (x, y, z, intensity,
    velocity, reflectivity, time_offset_ns, sensor_id, vx, vy, vz); StandardE2E
    keeps only xyz (the ``LidarComponent`` set). Points are in the Aeva
    reference frame and must be transformed to the ego frame by the caller.
    """
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if raw.size % _AEVA_NUM_COLS != 0:
        raise ValueError(
            f"{path}: {raw.size} float64 values not divisible by "
            f"{_AEVA_NUM_COLS} Aeva columns"
        )
    return raw.reshape(-1, _AEVA_NUM_COLS)[:, :3].astype(np.float32)


def read_boxes(path: str) -> list[dict[str, Any]]:
    """Read a per-frame 3D bounding-box JSON (a list of object dicts)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [obj for obj in data if isinstance(obj, dict)]
