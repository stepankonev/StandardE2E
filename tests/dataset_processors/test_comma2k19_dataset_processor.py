# flake8: noqa: E501
"""Tests for the comma2k19 processor.

* **Coordinate math** (always runs): the quaternion->rotation convention
  (``ecef_from_device``), the FRD<->FLU involution, the assembled
  ``T_world_ego`` for a known orientation, and the ECEF->geodetic helper on a
  hard-coded fix. Plus directory discovery / segment-id formatting on a
  synthetic layout.
* **Real-frame checks** (skipped unless ``COMMA2K19_ROOT`` points at extracted
  segments or ``Chunk_*.zip`` archives): a built frame has the single FRONT
  camera with pinhole intrinsics and identity extrinsics, a speed that matches
  the ECEF velocity, a rigid ego pose whose FRD ``down`` axis aligns with
  gravity, and a forward-only video reader that decodes deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.src_datasets.comma2k19._comma_geometry import (
    FRD_TO_FLU,
    ecef_to_geodetic,
    pose_matrices_world_from_ego,
    quats_to_rotmats,
)
from standard_e2e.caching.src_datasets.comma2k19._comma_io import (
    discover_segments,
    load_pose_arrays,
)
from standard_e2e.caching.src_datasets.comma2k19.comma2k19_dataset_processor import (
    Comma2k19DatasetProcessor,
)
from standard_e2e.enums import CameraDirection
from standard_e2e.enums import TrajectoryComponent as TC

# Real-frame checks read comma2k19 only from this env var; they skip when unset.
_COMMA2K19_ROOT = os.environ.get("COMMA2K19_ROOT", "")


# --------------------------------------------------------------------------- #
# Coordinate-math unit tests (no data required)
# --------------------------------------------------------------------------- #
def test_quats_to_rotmats_identity_and_known_rotation():
    # Identity quaternion -> identity matrix.
    r = quats_to_rotmats(np.array([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(r[0], np.eye(3), atol=1e-12)
    # +90 deg about z (Hamilton w,x,y,z = cos45, 0, 0, sin45): active rotation
    # mapping device +x -> +y.
    s = np.sqrt(0.5)
    r = quats_to_rotmats(np.array([[s, 0.0, 0.0, s]]))[0]
    np.testing.assert_allclose(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    # Always a proper rotation.
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_quats_to_rotmats_normalises_and_validates():
    # Non-unit input is normalised.
    r = quats_to_rotmats(np.array([[2.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(r[0], np.eye(3), atol=1e-12)
    with pytest.raises(ValueError):
        quats_to_rotmats(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        quats_to_rotmats(np.zeros((1, 4)))  # zero-norm quaternion


def test_frd_to_flu_is_an_involution_negating_y_and_z():
    np.testing.assert_allclose(FRD_TO_FLU @ FRD_TO_FLU, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(FRD_TO_FLU @ [1, 0, 0], [1, 0, 0])  # forward kept
    np.testing.assert_allclose(FRD_TO_FLU @ [0, 1, 0], [0, -1, 0])  # right -> -left
    np.testing.assert_allclose(FRD_TO_FLU @ [0, 0, 1], [0, 0, -1])  # down -> -up
    assert np.isclose(np.linalg.det(FRD_TO_FLU), 1.0)


def test_pose_world_from_ego_identity_orientation():
    # Identity device orientation -> ecef_from_device = I, so ego-FLU axes map
    # to world via diag(1, -1, -1). Origin defaults to the first position.
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    t = pose_matrices_world_from_ego(positions, quats)
    np.testing.assert_allclose(t[0, :3, 3], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(t[1, :3, 3], [10, 0, 0], atol=1e-12)
    np.testing.assert_allclose(t[0, :3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-12)
    # ego forward -> world +x; ego up -> world -z under the identity case.
    np.testing.assert_allclose(t[0, :3, :3] @ [1, 0, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(t[0, :3, :3] @ [0, 0, 1], [0, 0, -1], atol=1e-12)
    assert np.isclose(np.linalg.det(t[0, :3, :3]), 1.0)


def test_pose_world_from_ego_origin_shift():
    positions = np.array([[100.0, 200.0, 300.0], [101.0, 200.0, 300.0]])
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    t = pose_matrices_world_from_ego(positions, quats, origin_ecef=positions[0])
    np.testing.assert_allclose(t[0, :3, 3], [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(t[1, :3, 3], [1, 0, 0], atol=1e-9)


def test_ecef_to_geodetic_known_fix():
    # ECEF of a real comma2k19 segment origin (San Francisco Bay Area) -> the
    # latitude/longitude its GNSS log reports, to ~1 m.
    lat, lon, alt = ecef_to_geodetic([-2713181.4457, -4266758.2629, 3874739.1173])
    assert np.isclose(lat, 37.64938, atol=1e-3)
    assert np.isclose(lon, -122.45180, atol=1e-3)
    assert -200.0 < alt < 1000.0


def test_discover_dir_segments_and_segment_id(tmp_path):
    seg = tmp_path / "Chunk_1" / "abc123def456|2018-01-02--03-04-05" / "7"
    (seg / "global_pose").mkdir(parents=True)
    (seg / "global_pose" / "frame_positions").write_bytes(b"\x00")
    (seg / "video.hevc").write_bytes(b"\x00")
    segments = discover_segments(str(tmp_path))
    assert len(segments) == 1
    ref = segments[0]
    assert ref.path == str(seg)
    assert ref.segment_id == "abc123def456_2018-01-02--03-04-05_7"
    assert ref.route == "abc123def456|2018-01-02--03-04-05"
    assert ref.segment == "7"


def test_discover_segments_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_segments(str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        discover_segments(str(tmp_path))  # exists but holds no segments


def test_camera_intrinsics_are_eon_pinhole(tmp_path):
    proc = Comma2k19DatasetProcessor(
        common_output_path=str(tmp_path), split="all", context_aggregators=[]
    )
    k = proc.camera_intrinsics
    assert k.shape == (3, 3) and k.dtype == np.float32
    assert np.isclose(k[0, 0], 910.0) and np.isclose(k[1, 1], 910.0)
    assert np.isclose(k[0, 2], 1164 / 2) and np.isclose(k[1, 2], 874 / 2)


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
def _first_segment():
    if not _COMMA2K19_ROOT or not Path(_COMMA2K19_ROOT).exists():
        return None
    try:
        segments = discover_segments(_COMMA2K19_ROOT)
    except FileNotFoundError:
        return None
    return segments[0] if segments else None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ref = _first_segment()
    if ref is None:
        pytest.skip("no comma2k19 data available (set COMMA2K19_ROOT)")
    out = tmp_path_factory.mktemp("comma2k19")
    proc = Comma2k19DatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    poses = load_pose_arrays(ref)
    mid = len(poses["frame_times"]) // 2
    fd = proc._prepare_standardized_frame_data((ref, mid))
    yield proc, ref, poses, mid, fd
    proc.cleanup()


def test_frame_has_single_front_camera(built):
    _proc, _ref, _poses, _mid, fd = built
    assert set(fd.cameras) == {CameraDirection.FRONT}
    cam = fd.cameras[CameraDirection.FRONT]
    assert cam.image.ndim == 3 and cam.image.dtype == np.uint8
    assert cam.image.shape == (874, 1164, 3)
    assert cam.intrinsics.shape == (3, 3)
    assert cam.extrinsics.shape == (4, 4)
    np.testing.assert_allclose(cam.extrinsics, np.eye(4), atol=1e-6)
    assert not cam.is_fisheye and cam.distortion is None


def test_global_position_speed_matches_velocity(built):
    _proc, _ref, poses, mid, fd = built
    speed = float(fd.global_position.get(TC.SPEED)[0, 0])
    expected = float(np.linalg.norm(poses["frame_velocities"][mid]))
    assert np.isclose(speed, expected, atol=1e-3)
    assert speed >= 0.0


def test_ego_pose_is_rigid_and_finite(built):
    _proc, _ref, _poses, _mid, fd = built
    t = fd.aux_data["pose_matrix"]
    assert t.shape == (4, 4)
    assert np.isfinite(t).all()
    np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=1e-6)


def test_frd_down_axis_aligns_with_gravity(built):
    """Regression guard on the ecef_from_device convention: the device 'down'
    axis must point toward the Earth centre (the reading that yields a
    physically sane trajectory; the transpose does not)."""
    _proc, _ref, poses, mid, _fd = built
    r_ecef_frd = quats_to_rotmats(poses["frame_orientations"][mid : mid + 1])[0]
    down_in_ecef = r_ecef_frd[:, 2]  # device z (down) expressed in ECEF
    toward_center = -poses["frame_positions"][mid]
    toward_center = toward_center / np.linalg.norm(toward_center)
    assert float(down_in_ecef @ toward_center) > 0.95


def test_forward_only_reader_is_deterministic(built):
    proc, _ref, poses, mid, _fd = built
    n = len(poses["frame_times"])
    img_a = proc._read_frame(mid)
    img_b = proc._read_frame(mid)  # re-read (reopens; same frame)
    assert np.array_equal(img_a, img_b)
    other = proc._read_frame(min(mid + 5, n - 1))
    assert not np.array_equal(img_a, other)


def test_image_max_size_downscales_and_scales_intrinsics(tmp_path_factory):
    ref = _first_segment()
    if ref is None:
        pytest.skip("no comma2k19 data available (set COMMA2K19_ROOT)")
    out = tmp_path_factory.mktemp("comma2k19_lowres")
    mid = len(load_pose_arrays(ref)["frame_times"]) // 2
    full = Comma2k19DatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    small = Comma2k19DatasetProcessor(
        common_output_path=str(out),
        split="all",
        context_aggregators=[],
        image_max_size=300,
    )
    cam_full = full._prepare_standardized_frame_data((ref, mid)).cameras[
        CameraDirection.FRONT
    ]
    cam_small = small._prepare_standardized_frame_data((ref, mid)).cameras[
        CameraDirection.FRONT
    ]
    full.cleanup()
    small.cleanup()
    assert max(cam_full.image.shape[:2]) == 1164  # native, unchanged
    assert max(cam_small.image.shape[:2]) <= 300
    # Intrinsics scale by the same per-axis ratio as the image (so projection
    # still holds at the reduced resolution).
    sx = cam_small.image.shape[1] / cam_full.image.shape[1]
    sy = cam_small.image.shape[0] / cam_full.image.shape[0]
    np.testing.assert_allclose(
        cam_small.intrinsics[0, 0], cam_full.intrinsics[0, 0] * sx, rtol=1e-3
    )
    np.testing.assert_allclose(
        cam_small.intrinsics[1, 1], cam_full.intrinsics[1, 1] * sy, rtol=1e-3
    )
    np.testing.assert_allclose(
        cam_small.intrinsics[0, 2], cam_full.intrinsics[0, 2] * sx, rtol=1e-3
    )
    np.testing.assert_allclose(
        cam_small.intrinsics[1, 2], cam_full.intrinsics[1, 2] * sy, rtol=1e-3
    )
