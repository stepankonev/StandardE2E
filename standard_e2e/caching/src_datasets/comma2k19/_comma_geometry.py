"""Pose geometry helpers for comma2k19.

comma2k19 stores, per video frame, a ``global_pose`` directory of NumPy arrays
(``frame_positions``, ``frame_orientations``, ``frame_velocities``,
``frame_times``). The pose of the road camera is given as:

* ``frame_positions``    -- (N, 3) camera position in ECEF metres.
* ``frame_orientations`` -- (N, 4) Hamilton quaternion ``(w, x, y, z)``.
* ``frame_velocities``   -- (N, 3) camera velocity in ECEF metres/second.

The dataset README describes the quaternion as the orientation "from ECEF to
local frame [forward, right, down]". The rotation *matrix* that comma's own
``rotations_from_quats`` builds from that quaternion is, however,
``ecef_from_device`` -- it maps a vector from the device frame (FRD:
x-forward, y-right, z-down) **into** ECEF. We verified this direction
empirically against the data (see ``test_comma2k19_dataset_processor.py``):
the FRD ``down`` axis lines up with gravity (toward the Earth centre) and the
resulting per-frame trajectory is physically sane (forward motion, ~2% road
grade) only under the device->ECEF reading; the transpose puts hundreds of
metres of bogus climb into one minute of driving.

StandardE2E's ego frame is FLU (x-forward, y-left, z-up) -- the same
convention used by the lidar / WayveScenes code and by the yaw extraction in
``FuturePastStatesFromMatricesAggregator``. ``FRD_TO_FLU`` converts between
FRD and FLU; ``pose_matrices_world_from_ego`` assembles the per-frame
``T_world_ego`` (ego-FLU -> world) used both for ``global_position`` and as
``aux_data["pose_matrix"]``.

The world frame is ECEF-axis-aligned with its origin shifted to a per-segment
anchor (the segment's first frame by default). The shift keeps within-segment
coordinates O(1 km) so they survive the float32 round-trip in ``Trajectory``;
it does not affect the future/past ego-relative trajectories, which are
invariant to the choice of (rigid) world frame.
"""

from __future__ import annotations

import numpy as np

from standard_e2e.utils.geometry import quats_wxyz_to_rotmats

# FRD (x-fwd, y-right, z-down) <-> FLU (x-fwd, y-left, z-up): negate y and z.
# The matrix is its own inverse, so it converts vectors in both directions.
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


# comma2k19 stores Hamilton (w, x, y, z) quaternions; the resulting matrix is
# ``ecef_from_device`` -- the device-FRD axes expressed as ECEF columns.
quats_to_rotmats = quats_wxyz_to_rotmats


def pose_matrices_world_from_ego(
    positions_ecef: np.ndarray,
    quats_wxyz: np.ndarray,
    origin_ecef: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble per-frame ``T_world_ego`` transforms (ego-FLU -> world).

    Args:
        positions_ecef: ``(N, 3)`` ECEF positions (metres).
        quats_wxyz: ``(N, 4)`` Hamilton quaternions ``(w, x, y, z)``.
        origin_ecef: ``(3,)`` world-frame origin in ECEF. Defaults to the
            first position so within-segment coordinates stay small.

    Returns:
        ``(N, 4, 4)`` float64 homogeneous transforms mapping ego-FLU points to
        the per-segment local world frame (ECEF axes, origin-shifted).
    """
    positions = np.asarray(positions_ecef, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}")
    r_ecef_frd = quats_to_rotmats(quats_wxyz)  # device(FRD) -> ECEF
    if r_ecef_frd.shape[0] != positions.shape[0]:
        raise ValueError("positions and quaternions must have the same length")
    r_world_ego = r_ecef_frd @ FRD_TO_FLU  # ego(FLU) -> world(ECEF axes)
    if origin_ecef is None:
        origin = positions[0]
    else:
        origin = np.asarray(origin_ecef, dtype=np.float64)
    n = positions.shape[0]
    transforms = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    transforms[:, :3, :3] = r_world_ego
    transforms[:, :3, 3] = positions - origin
    return transforms


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    """Convert a single ECEF position to WGS84 geodetic coordinates.

    Closed-form Bowring iteration. Used only for sanity checks against the
    dataset's GNSS fixes (see the tests); not on the preprocessing hot path.

    Args:
        xyz: ``(3,)`` ECEF position in metres.

    Returns:
        ``(latitude_deg, longitude_deg, altitude_m)``.
    """
    x, y, z = (float(v) for v in np.asarray(xyz, dtype=np.float64))
    a = 6378137.0  # WGS84 semi-major axis
    e2 = 6.69437999014e-3  # WGS84 first eccentricity squared
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    alt = 0.0
    for _ in range(8):
        s = np.sin(lat)
        n = a / np.sqrt(1.0 - e2 * s * s)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - e2 * n / (n + alt)))
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)
