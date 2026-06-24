"""Fixed-rig camera calibration for KITScenes LongTail.

LongTail ships no per-scenario calibration files; the six ring cameras share a
single fixed rig whose pinhole ``K`` and ``(R, t)`` extrinsics are published in
the dataset README (and only there). ``R`` is the ego->camera rotation and ``t``
the matching translation (``x_cam = R @ x_ego + t``) -- verified by the front
camera's optical axis (``R[2]``) landing on ego +x (forward). StandardE2E's
``CameraData.extrinsics`` is ``T_ego_from_camera`` (camera -> ego), so we store
``inv(se3(R, t))``.

The intrinsics correspond to the **processed** (non-``_raw``) image resolution
(3504x2272; principal point ~1765/1139); the ``_raw`` splits ship the native
3200x2200 frames, for which these intrinsics are only approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from standard_e2e.enums import CameraDirection
from standard_e2e.utils import se3

# Six ring-camera column/keys -> canonical CameraDirection (same surround set as
# KITScenes Multimodal). LongTail's parquet columns are ``frames_camera_<key>``.
CAMERA_TO_DIRECTION: dict[str, CameraDirection] = {
    "front": CameraDirection.FRONT,
    "front_left": CameraDirection.FRONT_LEFT,
    "front_right": CameraDirection.FRONT_RIGHT,
    "rear": CameraDirection.REAR,
    "rear_left": CameraDirection.REAR_LEFT,
    "rear_right": CameraDirection.REAR_RIGHT,
}

# Vendored verbatim from the KITScenes-LongTail dataset README (pinhole model).
_CAMERA_PARAMETERS: dict[str, dict[str, list]] = {
    "front": {
        "K": [[1841.0, 0.0, 1765.0], [0.0, 1841.0, 1139.0], [0.0, 0.0, 1.0]],
        "R": [
            [0.01709344, -0.99983669, 0.00586657],
            [0.00538969, -0.0057752, -0.9999688],
            [0.99983937, 0.01712452, 0.00529009],
        ],
        "t": [-0.0183397, -0.18646863, -0.20817565],
    },
    "front_left": {
        "K": [[1844.0, 0.0, 1764.0], [0.0, 1844.0, 1131.0], [0.0, 0.0, 1.0]],
        "R": [
            [0.87490947, -0.48422726, 0.00757555],
            [0.00382071, -0.00874058, -0.9999545],
            [0.48427144, 0.87489861, -0.00579713],
        ],
        "t": [-0.00990917, -0.18619818, -0.19092926],
    },
    "front_right": {
        "K": [[1845.0, 0.0, 1749.0], [0.0, 1845.0, 1138.0], [0.0, 0.0, 1.0]],
        "R": [
            [-8.51906736e-01, -5.23693303e-01, 4.85398655e-04],
            [1.04631263e-03, -2.62893385e-03, -9.99995998e-01],
            [5.23692486e-01, -8.51902824e-01, 2.78755405e-03],
        ],
        "t": [-0.00863543, -0.18592461, -0.22478478],
    },
    "rear": {
        "K": [[1845.0, 0.0, 1765.0], [0.0, 1845.0, 1135.0], [0.0, 0.0, 1.0]],
        "R": [
            [-2.32531565e-02, 9.99699927e-01, -7.70433147e-03],
            [-9.34666883e-04, -7.72815425e-03, -9.99969706e-01],
            [-9.99729168e-01, -2.32452442e-02, 1.11409296e-03],
        ],
        "t": [0.01893959, -0.18635925, -0.20259226],
    },
    "rear_left": {
        "K": [[1843.0, 0.0, 1769.0], [0.0, 1843.0, 1136.0], [0.0, 0.0, 1.0]],
        "R": [
            [0.85288016, 0.52210461, 0.00148434],
            [0.00680007, -0.00826537, -0.99994272],
            [-0.52206244, 0.85284141, -0.01059972],
        ],
        "t": [0.00806251, -0.18614501, -0.18860823],
    },
    "rear_right": {
        "K": [[1847.0, 0.0, 1756.0], [0.0, 1847.0, 1148.0], [0.0, 0.0, 1.0]],
        "R": [
            [-8.81148667e-01, 4.72814031e-01, 4.89123238e-03],
            [-4.20872288e-03, 2.50132715e-03, -9.99988016e-01],
            [-4.72820592e-01, -8.81158694e-01, -2.14095393e-04],
        ],
        "t": [0.01272224, -0.18561945, -0.22249366],
    },
}


@dataclass(frozen=True)
class LongTailCameraCalibration:
    """One ring camera's pinhole ``K`` + ``T_ego_from_camera`` extrinsics."""

    camera_key: str
    direction: CameraDirection
    intrinsics: np.ndarray  # (3, 3) float32
    extrinsics: np.ndarray  # (4, 4) float64, T_ego_from_camera


def build_calibrations() -> dict[CameraDirection, LongTailCameraCalibration]:
    """Build the fixed-rig calibration for the six ring cameras."""
    calibs: dict[CameraDirection, LongTailCameraCalibration] = {}
    for key, params in _CAMERA_PARAMETERS.items():
        direction = CAMERA_TO_DIRECTION[key]
        # README gives ego->camera (R, t); CameraData wants camera->ego.
        t_cam_from_ego = se3(np.asarray(params["R"]), np.asarray(params["t"]))
        calibs[direction] = LongTailCameraCalibration(
            camera_key=key,
            direction=direction,
            intrinsics=np.asarray(params["K"], dtype=np.float32),
            extrinsics=np.linalg.inv(t_cam_from_ego),
        )
    return calibs
