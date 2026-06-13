"""Calibration / pose / box geometry for View-of-Delft (KITTI-style frames).

VoD ships KITTI-format per-frame calibration (``calib/<id>.txt``) and a per-frame
ego pose (``pose/<id>.json``). StandardE2E expresses every per-frame sensor and
box in the **velodyne (LiDAR) frame** -- FLU (x-forward, y-left, z-up): the frame
the ``.bin`` point clouds already live in and the frame VoD defines box yaw
against. These helpers parse the raw text/JSON and reimplement the frame math
with numpy (no devkit dependency):

* :func:`parse_calibration` -> camera intrinsics ``K`` (3x3) and
  ``T_cam_from_lidar`` (the calib's ``Tr_velo_to_cam`` as a 4x4).
* :func:`ego_pose_map_from_lidar` -> ``T_map_from_lidar`` (the ego pose in the
  static map frame), composed from the pose JSON's ``mapToCamera`` and the calib.
* :func:`box_camera_to_ego` -> a KITTI label's camera-frame center + ``(h, w, l)``
  + ``rotation`` mapped to an ego-frame center, ``(length, width, height)`` and
  heading.

Box yaw: VoD defines ``rotation`` about the LiDAR's **negative** vertical (-Z)
axis (``docs/ANNOTATION.md``); the ego FLU heading (yaw about +Z) is therefore
``-rotation``. Box ``location`` is the KITTI **bottom-face** center (verified
empirically via lidar containment -- untreated, contained points pile up at
+H/2 inside the box), so :func:`box_camera_to_ego` raises it by H/2 along the
ego +z axis to the geometric center StandardE2E stores.
"""

from __future__ import annotations

import numpy as np

from standard_e2e.utils import transform_points

# Calibration line keys (KITTI format). ``P2`` is the (rectified) left-colour
# camera projection matrix; its left 3x3 block is the pinhole ``K``.
# ``Tr_velo_to_cam`` maps a velodyne-frame point into the camera frame.
_CALIB_INTRINSICS_KEY = "P2"
_CALIB_EXTRINSICS_KEY = "Tr_velo_to_cam"


def _row_major_3x4_to_4x4(values: np.ndarray) -> np.ndarray:
    """Pack 12 row-major ``[R | t]`` values into a 4x4 homogeneous transform."""
    if values.size != 12:
        raise ValueError(f"expected 12 values for a 3x4 transform; got {values.size}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = values.reshape(3, 4)
    return transform


def parse_calibration(calib_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a KITTI ``calib/<id>.txt`` body.

    Returns ``(intrinsics, t_cam_from_lidar)`` -- a ``(3, 3)`` float32 ``K`` and a
    ``(4, 4)`` float64 ``Tr_velo_to_cam``. Lines are parsed by key (``"P2: ..."``)
    so a trailing empty entry (e.g. ``Tr_imu_to_velo:``) is ignored.
    """
    values: dict[str, np.ndarray] = {}
    for raw_line in calib_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        numbers = rest.split()
        if not numbers:
            continue
        values[key.strip()] = np.array(numbers, dtype=np.float64)

    missing = [
        key
        for key in (_CALIB_INTRINSICS_KEY, _CALIB_EXTRINSICS_KEY)
        if key not in values
    ]
    if missing:
        raise ValueError(f"calibration missing {missing}; found keys {sorted(values)}")

    intrinsics = values[_CALIB_INTRINSICS_KEY].reshape(3, 4)[:3, :3].astype(np.float32)
    t_cam_from_lidar = _row_major_3x4_to_4x4(values[_CALIB_EXTRINSICS_KEY])
    return intrinsics, t_cam_from_lidar


def ego_pose_map_from_lidar(
    map_from_camera: np.ndarray, t_cam_from_lidar: np.ndarray
) -> np.ndarray:
    """Compose ``T_map_from_lidar`` = ``mapToCamera @ T_cam_from_lidar``.

    Despite its name, VoD's per-frame ``mapToCamera`` matrix is the **camera's
    pose in the map frame** (``T_map_from_camera``): its translation is the
    camera origin's map coordinates and changes smoothly frame-to-frame
    (empirically ~0.2 m between 10 Hz keyframes; the inverse gives ~50x too-fast,
    non-physical ego motion). Composing it with the static camera<-lidar calib
    yields the ego (lidar) pose in the map frame, as a ``(4, 4)`` float64
    transform mapping ego points into the map.
    """
    map_from_camera = np.asarray(map_from_camera, dtype=np.float64).reshape(4, 4)
    return map_from_camera @ np.asarray(t_cam_from_lidar, dtype=np.float64)


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` (numpy-backed)."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def box_camera_to_ego(
    location_cam: np.ndarray,
    dimensions_hwl: np.ndarray,
    rotation: float,
    t_lidar_from_cam: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Map one KITTI label box into the ego (lidar) frame.

    Args:
        location_cam: ``(3,)`` camera-frame box centre ``(x, y, z)``.
        dimensions_hwl: ``(3,)`` label dimensions ``(height, width, length)``.
        rotation: yaw about the LiDAR -Z axis (radians).
        t_lidar_from_cam: ``(4, 4)`` camera->lidar transform
            (``inv(Tr_velo_to_cam)``).

    Returns:
        ``(center_xyz, length, width, height, heading)`` with ``center_xyz`` a
        ``(3,)`` float64 ego-frame point and ``heading`` the FLU yaw (about +Z),
        wrapped to ``[-pi, pi]``.
    """
    location_cam = np.asarray(location_cam, dtype=np.float64).reshape(1, 3)
    center = transform_points(
        np.asarray(t_lidar_from_cam, dtype=np.float64), location_cam
    )[0]
    height, width, length = (
        float(v) for v in np.asarray(dimensions_hwl, dtype=np.float64).reshape(3)
    )
    # KITTI ``location`` is the box bottom-face center (verified empirically by
    # lidar containment); raise it by H/2 along ego +z (up) to the geometric
    # center StandardE2E expects.
    center = center + np.array([0.0, 0.0, height / 2.0])
    heading = _wrap_to_pi(-float(rotation))
    return center, length, width, height, heading
