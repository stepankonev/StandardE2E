# flake8: noqa: E501
"""Tests for the WayveScenes101 processor.

* **Coordinate math** (always runs): the OpenCV->FLU point conversion and the
  ``world_from_cam`` conjugation, on synthetic inputs.
* **Real-frame checks** (skipped unless an extracted scene is present): a built
  frame has the 5 cameras, an ego-frame range-clipped point cloud, and a finite
  ego pose — i.e. the modalities line up with the lidar-consistent design.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.src_datasets.wayve_scenes.wayve_scenes_dataset_processor import (
    _FLU_RDF,
    WayveScenesDatasetProcessor,
    _opencv_points_to_flu,
    _world_from_cam_flu,
)
from standard_e2e.enums import CameraDirection

_WAYVE_ROOTS = [
    Path(p)
    for p in (
        os.environ.get("WAYVE_SCENES_ROOT", ""),
        "/mnt/bigdisk/datasets/wayve_scenes_unzipped",
    )
    if p
]


def _find_scene() -> Path | None:
    for root in _WAYVE_ROOTS:
        if not root.is_dir():
            continue
        for scene in sorted(root.glob("scene_*")):
            if (scene / "colmap_sparse" / "rig" / "images.bin").is_file():
                return scene
    return None


# --------------------------------------------------------------------------- #
# Coordinate-math unit tests (no data required)
# --------------------------------------------------------------------------- #
def test_opencv_points_to_flu_axis_mapping():
    # OpenCV RDF (x-right, y-down, z-forward) -> Wayve FLU (x-fwd, y-left, z-up).
    # A point 5 m straight ahead in OpenCV is (0, 0, 5) -> FLU (5, 0, 0).
    pts = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    flu = _opencv_points_to_flu(pts)
    np.testing.assert_allclose(flu[0], [5.0, 0.0, 0.0])  # forward
    np.testing.assert_allclose(flu[1], [0.0, -1.0, 0.0])  # right -> -y
    np.testing.assert_allclose(flu[2], [0.0, 0.0, -1.0])  # down -> -z


def test_flu_rdf_is_orthonormal_rotation():
    R = _FLU_RDF[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_world_from_cam_translation_equals_flu_camera_center():
    # Identity camera->world (cam_from_world = identity) sits at the origin; a
    # translated one's FLU ego position must equal the FLU of its center.
    qvec = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
    tvec = np.array([1.0, 2.0, 3.0])  # cam_from_world translation
    T = _world_from_cam_flu(qvec, tvec)
    # camera centre in OpenCV world = -R^T t = -t (R = I)
    center_opencv = -tvec
    expected = _opencv_points_to_flu(center_opencv[None])[0]
    np.testing.assert_allclose(T[:3, 3], expected, atol=1e-9)
    # rigid transform: rotation block orthonormal
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)


# --------------------------------------------------------------------------- #
# Real-frame checks (skipped without data)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built_frame(tmp_path_factory):
    scene = _find_scene()
    if scene is None:
        pytest.skip("no extracted WayveScenes scene available")
    out = tmp_path_factory.mktemp("wayve")
    proc = WayveScenesDatasetProcessor(
        common_output_path=str(out), split="test", context_aggregators=[]
    )
    proc._refresh_scene_cache(scene)
    ts = sorted(proc._ego_pose)[len(proc._ego_pose) // 2]
    return proc, proc._prepare_standardized_frame_data((str(scene), ts))


def test_frame_has_five_cameras(built_frame):
    _proc, fd = built_frame
    assert set(fd.cameras) == {
        CameraDirection.FRONT,
        CameraDirection.FRONT_LEFT,
        CameraDirection.FRONT_RIGHT,
        CameraDirection.REAR_LEFT,
        CameraDirection.REAR_RIGHT,
    }
    front = fd.cameras[CameraDirection.FRONT]
    assert front.image.ndim == 3 and front.image.dtype == np.uint8
    assert front.intrinsics.shape == (3, 3)
    assert front.extrinsics.shape == (4, 4)
    assert front.is_fisheye and front.distortion is not None


def test_lidar_in_ego_frame_within_range(built_frame):
    proc, fd = built_frame
    assert fd.lidar is not None
    pts = fd.lidar.points.to_numpy()
    assert list(fd.lidar.points.columns) == ["x", "y", "z"]
    assert len(pts) > 0
    # every point inside the configured range sphere
    assert np.all(np.linalg.norm(pts, axis=1) <= proc.LIDAR_MAX_RANGE_M + 1e-3)


def test_ego_pose_and_global_position_finite(built_frame):
    _proc, fd = built_frame
    T = fd.aux_data["pose_matrix"]
    assert T.shape == (4, 4)
    assert np.isfinite(T).all()
    assert fd.global_position is not None
