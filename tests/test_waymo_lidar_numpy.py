"""Parity tests: numpy Waymo lidar decode vs the upstream TF implementation.

The processor's worker hot path uses
:func:`standard_e2e.utils.waymo_lidar_numpy.numpy_convert_range_image_to_point_cloud`
instead of ``waymo_open_dataset.utils.frame_utils.convert_range_image_to_point_cloud``.
These tests guard the equivalence at the data level so future changes
can't silently drift the lidar output away from the canonical TF
implementation.

The frames here are synthesized in-memory (no real Waymo tfrecord fixture
needed): we build minimal ``WaymoFrame`` protos that exercise both the
"explicit ``beam_inclinations``" branch (TOP laser) and the "uniform
inclination from min/max" branch (side lasers), plus the per-pixel-pose
correction that only applies to TOP.
"""

from __future__ import annotations

import numpy as np
import pytest

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
    MatrixFloat,
    MatrixInt32,
)
from standard_e2e.utils.waymo_lidar_numpy import (
    _proto_floats_to_ndarray,
    _rotation_from_rpy,
    numpy_convert_range_image_to_point_cloud,
    numpy_parse_range_image_and_camera_projection,
)

# Per-laser shape: TOP is 64x2650; side lasers are 200x600 in real Waymo data.
# Tests use smaller dimensions to keep the synthetic frames fast.
_TOP = 1  # LaserName.TOP per Waymo proto enum
_SIDE_LASERS = [2, 3, 4, 5]  # FRONT, SIDE_LEFT, SIDE_RIGHT, REAR


def _set_extrinsic(calib, transform_4x4: np.ndarray) -> None:
    calib.extrinsic.transform.extend(
        transform_4x4.astype(np.float64).flatten().tolist()
    )


def _make_range_image(rng: np.random.Generator, h: int, w: int) -> MatrixFloat:
    # [H, W, 4] = (range, intensity, elongation, no_label_zone).
    # Mix of positive ranges (valid points) and zeros (masked out).
    data = rng.uniform(0.0, 60.0, (h, w, 4)).astype(np.float32)
    # Mask ~30% of pixels to range=0 to exercise the mask path.
    mask = rng.uniform(size=(h, w)) < 0.3
    data[..., 0][mask] = 0.0
    ri = MatrixFloat()
    ri.shape.dims.extend([h, w, 4])
    ri.data.extend(data.flatten().tolist())
    return ri


def _make_top_pose(rng: np.random.Generator, h: int, w: int) -> MatrixFloat:
    # [H, W, 6] = (roll, pitch, yaw, tx, ty, tz) per pixel, tiny perturbations.
    data = rng.uniform(-0.005, 0.005, (h, w, 6)).astype(np.float32)
    top_pose = MatrixFloat()
    top_pose.shape.dims.extend([h, w, 6])
    top_pose.data.extend(data.flatten().tolist())
    return top_pose


def _build_synthetic_frame(seed: int = 0):
    """Return (frame, range_images, camera_projections, top_pose).

    Dimensions:
      - TOP laser: 16 x 100 (small enough to keep tests fast)
      - Side lasers: 8 x 60
    """
    rng = np.random.default_rng(seed)

    frame = WaymoFrame()
    # Frame pose: random rigid transform.
    yaw, pitch, roll = rng.uniform(-0.3, 0.3, 3)
    R = _rotation_from_rpy(
        np.array(roll, dtype=np.float32),
        np.array(pitch, dtype=np.float32),
        np.array(yaw, dtype=np.float32),
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = rng.uniform(-10, 10, 3)
    frame.pose.transform.extend(pose.flatten().tolist())

    h_top, w_top = 16, 100
    h_side, w_side = 8, 60

    range_images: dict[int, list[MatrixFloat]] = {}
    camera_projections: dict[int, list[MatrixInt32]] = {}

    # TOP laser: explicit beam_inclinations.
    calib_top = frame.context.laser_calibrations.add()
    calib_top.name = _TOP
    _set_extrinsic(calib_top, np.eye(4))
    calib_top.beam_inclinations.extend(
        np.linspace(-0.3, 0.3, h_top, dtype=np.float32).tolist()
    )
    range_images[_TOP] = [_make_range_image(rng, h_top, w_top)]
    cp_top = MatrixInt32()
    cp_top.shape.dims.extend([h_top, w_top, 6])
    cp_top.data.extend([0] * (h_top * w_top * 6))
    camera_projections[_TOP] = [cp_top]

    # Side lasers: empty beam_inclinations -> uniform inclination from min/max.
    for laser_id, dy in zip(_SIDE_LASERS, [0.0, 0.5, -0.5, 0.0]):
        calib = frame.context.laser_calibrations.add()
        calib.name = laser_id
        ext = np.eye(4, dtype=np.float64)
        ext[1, 3] = dy
        _set_extrinsic(calib, ext)
        calib.beam_inclination_min = -0.1
        calib.beam_inclination_max = 0.1
        range_images[laser_id] = [_make_range_image(rng, h_side, w_side)]
        cp = MatrixInt32()
        cp.shape.dims.extend([h_side, w_side, 6])
        cp.data.extend([0] * (h_side * w_side * 6))
        camera_projections[laser_id] = [cp]

    top_pose = _make_top_pose(rng, h_top, w_top)
    return frame, range_images, camera_projections, top_pose


def _tf_reference(frame, range_images, camera_projections, top_pose):
    """Call the upstream TF implementation; returns list of (N, 3) ndarrays."""
    pytest.importorskip("tensorflow")
    # Import inside the function so collection-time skip is honored.
    from standard_e2e.third_party.waymo_open_dataset.utils import frame_utils

    pts, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, top_pose
    )
    return pts


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


def test_rotation_from_rpy_matches_yaw_pitch_roll_composition():
    """``_rotation_from_rpy`` builds R = R_yaw @ R_pitch @ R_roll."""
    rng = np.random.default_rng(0)
    rpy = rng.uniform(-0.5, 0.5, (5, 3)).astype(np.float32)
    R = _rotation_from_rpy(rpy[..., 0], rpy[..., 1], rpy[..., 2])
    assert R.shape == (5, 3, 3)

    # Manual reference: R_yaw @ R_pitch @ R_roll applied to a unit x-vector
    # should produce identical result.
    for i in range(5):
        r, p, y = rpy[i].tolist()
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]],
            dtype=np.float32,
        )
        Ry = np.array(
            [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]],
            dtype=np.float32,
        )
        Rz = np.array(
            [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        expected = Rz @ Ry @ Rx
        np.testing.assert_allclose(R[i], expected, atol=1e-5)


def test_proto_floats_to_ndarray_preserves_values_and_dtype():
    rng = np.random.default_rng(1)
    values = rng.uniform(-100, 100, 100).astype(np.float32)
    ri = MatrixFloat()
    ri.data.extend(values.tolist())
    out = _proto_floats_to_ndarray(ri.data)
    assert out.dtype == np.float32
    assert out.shape == (100,)
    np.testing.assert_allclose(out, values, atol=1e-6)


def test_numpy_lidar_decode_matches_tf_on_synthetic_frame():
    """End-to-end parity: every laser is byte-exact or float-noise close vs TF."""
    pytest.importorskip("tensorflow")

    frame, range_images, camera_projections, top_pose = _build_synthetic_frame(seed=7)

    pts_tf = _tf_reference(frame, range_images, camera_projections, top_pose)
    pts_np = numpy_convert_range_image_to_point_cloud(frame, range_images, top_pose)

    assert len(pts_tf) == len(pts_np), "laser count differs"

    # Calibrations are sorted by laser.name. Index 0 is TOP; the rest are
    # side lasers in numeric-name order.
    for laser_idx, (a, b) in enumerate(zip(pts_tf, pts_np)):
        assert a.shape == b.shape, f"laser {laser_idx}: shape {a.shape} vs {b.shape}"
        if laser_idx == 0:
            # TOP laser goes through per-pixel pose + frame_pose inverse.
            # Float32 noise from ``inv`` differs slightly between
            # ``tf.linalg.inv`` and ``np.linalg.inv``; ~3 mm on real data.
            np.testing.assert_allclose(a, b, atol=1e-2)
        else:
            # Side lasers: bit-exact (no inverse, no per-pixel pose chain).
            np.testing.assert_allclose(a, b, atol=1e-5)


def test_numpy_lidar_decode_handles_all_zero_range():
    """Frame with all-masked-out range pixels yields empty point clouds."""
    frame, range_images, _, top_pose = _build_synthetic_frame(seed=2)
    # Force every pixel to range=0.
    for laser_id, ris in range_images.items():
        ri = ris[0]
        dims = list(ri.shape.dims)
        ri.ClearField("data")
        ri.data.extend([0.0] * int(np.prod(dims)))

    pts = numpy_convert_range_image_to_point_cloud(frame, range_images, top_pose)
    assert len(pts) == 5
    for arr in pts:
        assert arr.shape == (0, 3)
        assert arr.dtype == np.float32


def test_numpy_parse_handles_empty_returns():
    """``numpy_parse_range_image_and_camera_projection`` on a frame with no
    laser returns empty dicts; mirrors the upstream behavior shape-wise."""
    frame = WaymoFrame()  # no lasers populated
    range_images, cps, sls, top_pose = numpy_parse_range_image_and_camera_projection(
        frame
    )
    assert range_images == {}
    assert cps == {}
    assert sls == {}
    # top_pose stays a default-constructed MatrixFloat (empty .data).
    assert len(top_pose.data) == 0
