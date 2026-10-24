"""Sensor geometry and timing of the MUHOS rig, **estimated by StandardE2E from the
data**.

MUHOS publishes no calibration: the recording kept the image and point-cloud topics but
not ``camera_info`` or ``tf``. The values below were fitted on the released data
(method and residuals in ``datasets/muhos/README.md``, *Calibration*). The rig did not
move between the recording days (the car's own body sits at the same pixels and lidar
coordinates in every set). Treat the values as a good estimate, not as the authors'
calibration.

Frames
* **lidar** (Hesai OT128 driver frame): x forward, y left, z up, mounted on the roof
  pitched ~6.7 deg nose-down.
* **ego** (StandardE2E): the lidar frame levelled -- z along the road-surface normal,
  x along the direction of travel on straight road -- with its origin on the road
  surface directly below the lidar. x forward, y left, z up, metres.
* **left** / **right**: the MultiSense S27 rectified stereo pair (pinhole, OpenCV
  optical frame, a common ``K``, the right camera ``BASELINE_M`` along the left
  camera's x axis).
* **aux**: the S27 colour camera (unrectified wide-angle lens, OpenCV Kannala-Brandt
  model), the image StandardE2E stores as ``CameraDirection.FRONT``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from standard_e2e.utils import intrinsics_matrix, se3

# --- lidar -> ego -----------------------------------------------------------------
# Road-surface normal in the lidar frame: median over the seven sets of RANSAC
# ground-plane fits (200 sweeps per set; nose-down pitch 6.53-6.94 deg per set, median
# 6.73 deg; roll median -0.02 deg); the direction of travel in the lidar frame
# (sweep-to-sweep ICP on straight road); the lidar's height above the road surface
# (median of the set medians).
GROUND_NORMAL_LIDAR = np.array([-0.117139, -0.000260, 0.993116])
TRAVEL_DIRECTION_LIDAR = np.array([0.99382, -0.00350, 0.11094])
LIDAR_HEIGHT_M = 1.999


def _ego_from_lidar() -> np.ndarray:
    z = GROUND_NORMAL_LIDAR / np.linalg.norm(GROUND_NORMAL_LIDAR)
    x = TRAVEL_DIRECTION_LIDAR - TRAVEL_DIRECTION_LIDAR @ z * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])  # rows: ego axes in lidar coordinates
    return se3(rotation, np.array([0.0, 0.0, LIDAR_HEIGHT_M]))


T_EGO_FROM_LIDAR = _ego_from_lidar()

# The car's own body (hood 1.3-2.1 m ahead of the lidar at ~1.0 m, roof behind it at
# ~1.7 m) returns ~1700 points per sweep; points inside this ego-frame box are dropped.
# Bounds (x, y, z) in metres; a 0.6 m shell around it holds no points.
EGO_BODY_BOX = ((-2.6, 2.6), (-1.0, 1.0), (0.1, 2.3))

# --- stereo pair (rectified) ------------------------------------------------------
# At standstill, lidar depth x SGBM disparity gives fx * B = 158.3 px m; the S27's
# nominal baseline fits best (rigid stereo-vs-lidar ICP residuals are smallest at
# 0.265-0.27 m).
STEREO_IMAGE_SIZE = (600, 960)  # (H, W)
BASELINE_M = 0.27
STEREO_FX = 586.2
STEREO_CX, STEREO_CY = 480.0, 300.0
STEREO_INTRINSICS = intrinsics_matrix(STEREO_FX, STEREO_FX, STEREO_CX, STEREO_CY)
# Rigid point-to-plane ICP of the metric stereo cloud onto the lidar at standstill (no
# motion, no timing), mean of the city_center and 60kmh sets (they agree to 0.07 deg and
# 5 mm).
T_LIDAR_FROM_LEFT = se3(
    np.array(
        [
            [0.006198036, -0.160253554, 0.987056424],
            [-0.999980403, -0.001863768, 0.005976598],
            [0.000881873, -0.987074123, -0.160261965],
        ]
    ),
    np.array([0.248323, 0.144410, -0.174424]),
)
T_LEFT_FROM_RIGHT = se3(np.eye(3), np.array([BASELINE_M, 0.0, 0.0]))

# --- aux colour camera ------------------------------------------------------------
# Dense fit at standstill: the colour image warped into the left view through the model
# (3D points from the metric stereo depth), DIS optical flow for the residual, model
# refit; 0.15 px median residual on held-out frames. Square pixels (the fitted aspect is
# 0.9999).
AUX_IMAGE_SIZE = (594, 960)  # (H, W)
AUX_INTRINSICS = intrinsics_matrix(408.6915, 408.6549, 474.7636, 304.8132)
AUX_DISTORTION = np.array([-0.025214, 0.052710, -0.160023, 0.128518])  # k1..k4
T_AUX_FROM_LEFT = se3(
    Rotation.from_rotvec([-0.001503397, 0.000627586, 0.001328233]).as_matrix(),
    np.array([-0.033768, -0.000736, 0.005077]),
)
T_EGO_FROM_LEFT = T_EGO_FROM_LIDAR @ T_LIDAR_FROM_LEFT
T_EGO_FROM_AUX = T_EGO_FROM_LEFT @ np.linalg.inv(T_AUX_FROM_LEFT)
T_EGO_FROM_RIGHT = T_EGO_FROM_LEFT @ T_LEFT_FROM_RIGHT

# --- timing -------------------------------------------------------------------------
# The cameras expose an image this long before its stamp, on the lidar's clock: moving
# frames' stereo clouds match the lidar moved to (stamp - CAMERA_DELAY_S); at the stamp
# the along-track offset grows linearly with speed: 14 ms (+-2 ms) on the highway set.
CAMERA_DELAY_S = 0.014
