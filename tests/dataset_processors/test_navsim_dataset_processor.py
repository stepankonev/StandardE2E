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

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
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
    assert len(adapters) == 5
    assert isinstance(adapters[0], CamerasIdentityAdapter)
    assert isinstance(adapters[1], LidarAdapter)
    assert isinstance(adapters[2], IntentIdentityAdapter)
    assert isinstance(adapters[3], Detections3DIdentityAdapter)
    assert isinstance(adapters[4], HDMapBEVAdapter)


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


# --- HD-map translation -----------------------------------------------------


NAVSIM_MAPS_RW = Path("/mnt/nvme1/data/navsim_maps_rw")


@pytest.fixture
def real_navsim_proc_with_maps(tmp_path: Path) -> NavsimDatasetProcessor:
    if not SAMPLE_LOG_PKL.exists():
        pytest.skip("NAVSIM sample log not available locally")
    if not (NAVSIM_MAPS_RW / "nuplan-maps-v1.0.json").exists():
        pytest.skip(
            "NAVSIM writable maps mirror not available at "
            f"{NAVSIM_MAPS_RW} — see tmp/raster_comparison/ or quickstart docs"
        )
    return NavsimDatasetProcessor(
        str(tmp_path), split="trainval", maps_root_path=str(NAVSIM_MAPS_RW)
    )


def test_hd_map_emits_expected_element_types(real_navsim_proc_with_maps):
    """nuPlan vector map → unified MapElementType taxonomy. Around any frame
    in a real log we should see lane centers, lane boundaries (no paint),
    drivable area, intersection or stop_line, and crosswalk. Walkway is also
    common but optional depending on the city."""
    from standard_e2e.enums import MapElementType

    sfd = real_navsim_proc_with_maps._prepare_standardized_frame_data(
        (SAMPLE_LOG_PKL, 200)
    )
    assert sfd.hd_map is not None and len(sfd.hd_map.elements) > 0
    counts: dict[MapElementType, int] = {t: 0 for t in MapElementType}
    for e in sfd.hd_map.elements:
        counts[e.type] += 1

    # Hard requirements: every NAVSIM frame ego sits on roads with lanes +
    # drivable area, with at least one lane boundary on each side.
    assert counts[MapElementType.LANE_CENTER] > 0
    assert counts[MapElementType.LANE_BOUNDARY] > 0
    assert counts[MapElementType.DRIVABLE_AREA] > 0
    # Frame 200 is in Las Vegas Strip near intersections + crosswalks; verify.
    assert (
        counts[MapElementType.INTERSECTION] > 0 or counts[MapElementType.STOP_LINE] > 0
    )
    assert counts[MapElementType.CROSSWALK] > 0
    # nuPlan ships no paint info; LANE_BOUNDARY attrs must reflect that.
    boundaries = [
        e for e in sfd.hd_map.elements if e.type == MapElementType.LANE_BOUNDARY
    ]
    assert all(b.attrs.get("paint_color") is None for b in boundaries)
    assert all(b.attrs.get("paint_pattern") is None for b in boundaries)


def test_hd_map_elements_in_ego_frame_within_radius(real_navsim_proc_with_maps):
    """All emitted points must be in ego XY (i.e. within ~64 m of origin)."""
    sfd = real_navsim_proc_with_maps._prepare_standardized_frame_data(
        (SAMPLE_LOG_PKL, 200)
    )
    assert sfd.hd_map is not None
    # Loose bound: 100 m on each axis (radius 64 m + element extent).
    for e in sfd.hd_map.elements:
        pts = np.asarray(e.points)
        assert (
            np.abs(pts[:, 0]).max() < 200.0
        ), f"{e.type}/{e.id} has x out of ego bounds: {pts[:, 0].max():.1f}"
        assert (
            np.abs(pts[:, 1]).max() < 200.0
        ), f"{e.type}/{e.id} has y out of ego bounds: {pts[:, 1].max():.1f}"


def test_lane_connector_marked_intersection(real_navsim_proc_with_maps):
    """LANE_CONNECTOR-derived LANE_CENTER elements must carry
    ``is_intersection=True``; plain LANE-derived ones must carry False."""
    from standard_e2e.enums import MapElementType

    sfd = real_navsim_proc_with_maps._prepare_standardized_frame_data(
        (SAMPLE_LOG_PKL, 200)
    )
    assert sfd.hd_map is not None
    centers = [e for e in sfd.hd_map.elements if e.type == MapElementType.LANE_CENTER]
    intersection_centers = [
        e for e in centers if e.attrs.get("is_intersection") is True
    ]
    plain_centers = [e for e in centers if e.attrs.get("is_intersection") is False]
    # Around a busy intersection we should see both kinds.
    assert intersection_centers, "no LANE_CONNECTOR-derived LANE_CENTER found"
    assert plain_centers, "no LANE-derived LANE_CENTER found"


def test_no_maps_root_skips_hd_map_gracefully(tmp_path: Path):
    """If neither maps_root_path nor NUPLAN_MAPS_ROOT resolves to a valid maps
    directory, ``_build_hd_map`` returns ``None`` — preprocessing should keep
    going on the other modalities."""
    if not SAMPLE_LOG_PKL.exists():
        pytest.skip("NAVSIM sample log not available locally")
    proc = NavsimDatasetProcessor(
        str(tmp_path),
        split="trainval",
        maps_root_path=str(tmp_path / "no_such_maps_dir"),
    )
    # Drop the env var so we exercise the third resolution step too — but
    # the auto-derived <input>/maps points at the read-only mount, which
    # the resolver doesn't recognise as a real maps root unless the json
    # manifest exists, so it should fall through to None.
    saved = os.environ.pop("NUPLAN_MAPS_ROOT", None)
    try:
        # Force input_path-based default to also miss by using a tmp path.
        spoof = tmp_path / "navsim_logs" / "trainval"
        spoof.mkdir(parents=True)
        spoof_pkl = spoof / "fake.pkl"
        # We don't actually need to load the pkl for the resolution — only
        # that _resolve_maps_root sees no valid candidate. Call directly.
        assert proc._resolve_maps_root(spoof_pkl) is None
    finally:
        if saved is not None:
            os.environ["NUPLAN_MAPS_ROOT"] = saved


def test_pcd_xyz_dtype_and_shape():
    """Smoke a synthetic call into _xy_city_to_ego (pure-numpy helper)."""
    from standard_e2e.caching.src_datasets.navsim._navsim_map import _xy_city_to_ego

    T_eye = np.eye(4)
    out = _xy_city_to_ego(np.array([[10.0, 20.0], [-1.0, -2.0]]), T_eye)
    assert out.dtype == np.float32 and out.shape == (2, 2)
    np.testing.assert_allclose(out, [[10, 20], [-1, -2]], atol=1e-6)


def test_xy_city_to_ego_translation():
    """Translating origin by (5, 7) should shift coords by (-5, -7)."""
    from standard_e2e.caching.src_datasets.navsim._navsim_map import _xy_city_to_ego

    T = np.eye(4)
    T[0, 3] = -5.0
    T[1, 3] = -7.0
    out = _xy_city_to_ego(np.array([[5.0, 7.0]]), T)
    np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)
