"""Tests for ``NavsimDatasetProcessor``.

Three layers:

* **Construction defaults** — adapters / context aggregators / split list
  wired correctly. Always runs.
* **Pure-Python helpers** — driving-command decoding, the binary-PCD
  reader. Always runs.
* **End-to-end on a real pickle + sensor blobs** — wires the processor to
  one local NAVSIM frame and confirms the resulting ``StandardFrameData``
  carries the expected modalities. Skipped automatically when the local
  OpenScene-v1.1 mount is unavailable.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    IntentIdentityAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.navsim._pcd import read_navsim_pcd_xyz
from standard_e2e.caching.src_datasets.navsim.navsim_dataset_processor import (
    NavsimDatasetProcessor,
    _driving_command_to_intent,
)
from standard_e2e.enums import CameraDirection, Intent

NAVSIM_ROOT = Path("/mnt/bigdisk/datasets/navsim/openscene-v1.1")
SAMPLE_LOG_PKL = (
    NAVSIM_ROOT
    / "navsim_logs"
    / "trainval"
    / "2021.05.12.19.36.12_veh-35_00005_00204.pkl"
)


# --- construction defaults ---------------------------------------------------


def test_navsim_defaults(tmp_path: Path):
    proc = NavsimDatasetProcessor(str(tmp_path), split="trainval")
    assert proc.dataset_name == "navsim"
    assert proc.allowed_splits == ["trainval"]
    adapters = getattr(proc, "_adapters")
    assert len(adapters) == 4
    assert isinstance(adapters[0], CamerasIdentityAdapter)
    assert isinstance(adapters[1], LidarAdapter)
    assert isinstance(adapters[2], IntentIdentityAdapter)
    assert isinstance(adapters[3], Detections3DIdentityAdapter)


def test_navsim_default_aggregators(tmp_path: Path):
    proc = NavsimDatasetProcessor(str(tmp_path), split="trainval")
    aggregators = proc.context_aggregators
    assert len(aggregators) == 2
    assert isinstance(aggregators[0], FuturePastStatesFromMatricesAggregator)
    assert isinstance(aggregators[1], FutureDetectionsAggregator)


def test_navsim_invalid_split_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        NavsimDatasetProcessor(str(tmp_path), split="train")


# --- driving-command decoding ------------------------------------------------


def test_driving_command_one_hot_left():
    assert _driving_command_to_intent(np.array([1, 0, 0, 0])) is Intent.GO_LEFT


def test_driving_command_one_hot_straight():
    assert _driving_command_to_intent(np.array([0, 1, 0, 0])) is Intent.GO_STRAIGHT


def test_driving_command_one_hot_right():
    assert _driving_command_to_intent(np.array([0, 0, 1, 0])) is Intent.GO_RIGHT


def test_driving_command_one_hot_unknown():
    assert _driving_command_to_intent(np.array([0, 0, 0, 1])) is Intent.UNKNOWN


def test_driving_command_malformed_falls_back_to_unknown():
    assert _driving_command_to_intent(np.array([0, 0, 0, 0])) is Intent.UNKNOWN
    assert _driving_command_to_intent(np.array([1, 1, 0, 0])) is Intent.UNKNOWN
    assert _driving_command_to_intent(np.array([1, 0, 0])) is Intent.UNKNOWN


# --- PCD reader on a synthetic file -----------------------------------------


def test_pcd_reader_round_trips_xyz(tmp_path: Path):
    """Hand-craft a 3-point PCD with the canonical NAVSIM layout, parse it."""
    header = (
        b"# .PCD v0.7 - Point Cloud Data file format\n"
        b"VERSION 0.7\n"
        b"FIELDS x y z intensity lidar_info ring\n"
        b"SIZE 4 4 4 1 1 1\n"
        b"TYPE F F F U U U\n"
        b"COUNT 1 1 1 1 1 1\n"
        b"WIDTH 3\n"
        b"HEIGHT 1\n"
        b"VIEWPOINT 0 0 0 1 0 0 0\n"
        b"POINTS 3\n"
        b"DATA binary\n"
    )
    points = [
        (1.0, 2.0, 3.0, 7, 0, 1),
        (4.5, -1.5, 0.25, 12, 1, 5),
        (-9.0, 0.0, 11.0, 0, 0, 0),
    ]
    body = b"".join(struct.pack("<fffBBB", *p) for p in points)
    pcd_path = tmp_path / "test.pcd"
    pcd_path.write_bytes(header + body)

    xyz = read_navsim_pcd_xyz(pcd_path)
    assert xyz.shape == (3, 3)
    assert xyz.dtype == np.float32
    np.testing.assert_allclose(
        xyz,
        np.array(
            [[1.0, 2.0, 3.0], [4.5, -1.5, 0.25], [-9.0, 0.0, 11.0]], dtype=np.float32
        ),
    )


# --- end-to-end on a real local frame ---------------------------------------


@pytest.fixture
def real_navsim_proc(tmp_path: Path) -> NavsimDatasetProcessor:
    if not SAMPLE_LOG_PKL.exists():
        pytest.skip("NAVSIM sample log not available locally")
    return NavsimDatasetProcessor(str(tmp_path), split="trainval")


def test_prepare_standardized_frame_data_on_real_log(real_navsim_proc):
    sfd = real_navsim_proc._prepare_standardized_frame_data((SAMPLE_LOG_PKL, 0))
    # Per-frame identity
    assert sfd.dataset_name == "navsim"
    assert sfd.split == "trainval"
    assert sfd.timestamp > 0  # unix-microsecond → seconds, definitely positive
    assert "scene_token" in sfd.extra_index_data
    assert "frame_token" in sfd.extra_index_data
    # 8 cameras, all populated
    assert sfd.cameras is not None
    assert len(sfd.cameras) == 8
    # Cover one expected direction
    assert CameraDirection.FRONT in sfd.cameras
    front = sfd.cameras[CameraDirection.FRONT]
    assert front.image.ndim == 3 and front.image.dtype == np.uint8
    # Lidar
    assert sfd.lidar is not None
    assert sfd.lidar.points.shape[0] > 1000  # NAVSIM merged sweeps are dense
    assert list(sfd.lidar.points.columns) == ["x", "y", "z"]
    # Detections — first frame of this log has annotations
    assert sfd.frame_detections_3d is not None
    assert len(sfd.frame_detections_3d.detections) > 0
    # Intent comes from the 4-element one-hot driving_command
    assert sfd.intent in (
        Intent.GO_LEFT,
        Intent.GO_STRAIGHT,
        Intent.GO_RIGHT,
        Intent.UNKNOWN,
    )
    # Aux carries the 4×4 ego pose
    assert sfd.aux_data is not None and "pose_matrix" in sfd.aux_data
    assert sfd.aux_data["pose_matrix"].shape == (4, 4)


def test_per_log_cache_hits_only_on_first_frame(
    real_navsim_proc, tmp_path: Path  # noqa: ARG001
):
    """Loading two frames from the same pickle should not reload the file."""
    proc = real_navsim_proc
    # First call populates the cache.
    proc._prepare_standardized_frame_data((SAMPLE_LOG_PKL, 0))
    cached = proc._frames
    # Second call must not reinstantiate the list (same object reference).
    proc._prepare_standardized_frame_data((SAMPLE_LOG_PKL, 1))
    assert proc._frames is cached
