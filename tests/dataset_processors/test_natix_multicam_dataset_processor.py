# flake8: noqa: E501
"""Tests for the NATIX Multi-Camera processor.

* **Geometry / parsing units** (always run): the body-FLU -> optical camera
  conversion, compass-heading -> FLU-yaw conversion, the
  unreliable-heading hold, pose assembly, the local metric projection, the
  calibration parser (cm -> m, distortion reorder, facing-verified direction
  map) and the trip-insight helpers.
* **Synthetic-trip end-to-end** (always run; hermetic): a tiny trip written
  to ``tmp_path`` -- real mp4s (cv2-encoded, frame index encoded in the
  pixels), CSVs with ``na`` gaps and a slower rear stream -- is discovered,
  aligned and built into ``StandardFrameData``, verifying the 1-based
  ``frame_number`` -> decoded-frame correspondence, the per-camera
  timestamp matching, timezone localization and the emitted pose.
* **Real-data checks** (skipped unless ``NATIX_MULTICAM_ROOT`` points at a
  downloaded tree, e.g. the Hugging Face ``dataset-sample/``): built frames
  carry plausible intrinsics for the actual footage, a rigid pose, first-fix
  timestamps consistent with ``trip_insight.json``'s ``startEpochMs`` (the
  timezone regression guard) and a heading that matches the GPS displacement
  direction while moving (the compass-convention regression guard).
"""

from __future__ import annotations

import json
import os
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io
from standard_e2e.caching.src_datasets.natix_multicam._natix_geometry import (
    BODY_FROM_OPTICAL,
    DEVICE_NAME_TO_DIRECTION,
    LocalMetricProjection,
    folder_key_from_device_name,
    headings_rad_from_deg,
    parse_fixed_metadata,
    poses_world_from_xy_heading,
    resolve_headings,
)
from standard_e2e.caching.src_datasets.natix_multicam.natix_multicam_dataset_converter import (
    NatixMulticamDatasetConverter,
)
from standard_e2e.caching.src_datasets.natix_multicam.natix_multicam_dataset_processor import (
    NatixMulticamDatasetProcessor,
)
from standard_e2e.enums import CameraDirection
from standard_e2e.enums import TrajectoryComponent as TC

# Real-data checks read NATIX only from this env var; they skip when unset.
_NATIX_ROOT = os.environ.get("NATIX_MULTICAM_ROOT", "")


# --------------------------------------------------------------------------- #
# Geometry units (no data required)
# --------------------------------------------------------------------------- #
def test_body_from_optical_is_a_proper_rotation_mapping_the_lens_axis():
    r = BODY_FROM_OPTICAL
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)
    # optical z (out the lens) -> body x (forward), optical x (right) ->
    # body -y, optical y (down) -> body -z.
    np.testing.assert_allclose(r @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(r @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(r @ [0, 1, 0], [0, 0, -1], atol=1e-12)


def test_headings_rad_from_deg_compass_to_flu():
    # 0 = north -> +pi/2; 90 = east -> 0; 180 = south -> -pi/2; 270 = west -> pi.
    yaw = headings_rad_from_deg(np.array([0.0, 90.0, 180.0, 270.0]))
    difference = yaw - np.array([np.pi / 2, 0.0, -np.pi / 2, np.pi])
    np.testing.assert_allclose(
        np.arctan2(np.sin(difference), np.cos(difference)), 0.0, atol=1e-12
    )
    # Wrapped into [-pi, pi] whatever the input range.
    assert np.all(np.abs(headings_rad_from_deg(np.array([720.5, -450.0]))) <= np.pi)


def test_resolve_headings_holds_last_reliable_heading():
    headings = np.array([np.nan, 10.0, 0.0, 0.0, 30.0])
    speeds = np.array([5.0, 5.0, 0.0, np.nan, 5.0])
    resolved = resolve_headings(headings, speeds)
    # Leading unreliable takes the first reliable (10); the stationary /
    # missing-speed fixes hold it; the last moving fix updates to 30.
    np.testing.assert_allclose(resolved, [10.0, 10.0, 10.0, 10.0, 30.0])


def test_resolve_headings_without_any_reliable_heading_is_zero():
    resolved = resolve_headings(np.array([np.nan, 0.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(resolved, [0.0, 0.0])


def test_poses_world_from_xy_heading_known_pose():
    poses = poses_world_from_xy_heading(
        np.array([3.0]), np.array([4.0]), np.array([np.pi / 2])
    )
    assert poses.shape == (1, 4, 4)
    t = poses[0]
    np.testing.assert_allclose(t[:3, 3], [3.0, 4.0, 0.0], atol=1e-12)
    # Ego forward (+x) points along world +y at yaw pi/2.
    np.testing.assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.isclose(np.linalg.det(t[:3, :3]), 1.0)
    with pytest.raises(ValueError):
        poses_world_from_xy_heading(np.zeros(2), np.zeros(1), np.zeros(2))


def test_local_metric_projection_anchor_and_northward_metre():
    projection = LocalMetricProjection(46.0, 7.0)
    x, y = projection.to_local_xy(np.array([46.0, 46.001]), np.array([7.0, 7.0]))
    np.testing.assert_allclose([x[0], y[0]], [0.0, 0.0], atol=1e-6)
    # 1e-3 deg of latitude is ~111.2 m due north.
    assert np.isclose(x[1], 0.0, atol=1e-3)
    assert np.isclose(y[1], 111.2, atol=0.5)


# --------------------------------------------------------------------------- #
# Calibration parsing
# --------------------------------------------------------------------------- #
def _camera_entry(device_name: str, rotation: np.ndarray, **overrides) -> dict:
    entry = {
        "device_name": device_name,
        "tx": 182.0,
        "ty": 0.0,
        "tz": 130.0,
        "fx": 1600.0,
        "fy": 1600.0,
        "cx": 640.0,
        "cy": 480.0,
        "k1": -0.4,
        "k2": 0.3,
        "k3": 0.45,
        "p1": 0.0,
        "p2": 0.0,
    }
    for i in range(3):
        for j in range(3):
            entry[f"r{i + 1}{j + 1}"] = float(rotation[i, j])
    entry.update(overrides)
    return entry


def test_parse_fixed_metadata_front_camera_units_and_optical_frame():
    metadata = {
        "device_extrinsics": [
            _camera_entry("camera_front", np.eye(3)),
            {"device_name": "vx360", "tx": 244.0, "ty": -35.0, "tz": 70},
        ]
    }
    calibrations = parse_fixed_metadata(metadata)
    assert set(calibrations) == {CameraDirection.FRONT}
    calib = calibrations[CameraDirection.FRONT]
    assert calib.folder_key == "FRONT"
    # Centimetres -> metres.
    np.testing.assert_allclose(calib.extrinsics[:3, 3], [1.82, 0.0, 1.30], atol=1e-9)
    # Identity body rotation -> the optical z (lens) axis is ego +x.
    np.testing.assert_allclose(
        calib.extrinsics[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        calib.intrinsics,
        [[1600.0, 0, 640.0], [0, 1600.0, 480.0], [0, 0, 1]],
        atol=1e-6,
    )
    # NATIX ships k1,k2,k3,p1,p2 -> container order k1,k2,p1,p2,k3.
    np.testing.assert_allclose(calib.distortion, [-0.4, 0.3, 0.0, 0.0, 0.45])


def test_parse_fixed_metadata_repeater_facing_maps_to_rear_side():
    # The 4-camera trips' LEFT camera is the backward-left-facing repeater.
    rotation = np.array([[-0.848, -0.53, 0.0], [0.53, -0.848, 0.0], [0.0, 0.0, 1.0]])
    metadata = {"device_extrinsics": [_camera_entry("camera_left", rotation)]}
    calibrations = parse_fixed_metadata(metadata)
    assert set(calibrations) == {CameraDirection.REAR_LEFT}
    lens_in_ego = calibrations[CameraDirection.REAR_LEFT].extrinsics[:3, :3] @ [0, 0, 1]
    assert lens_in_ego[0] < 0 and lens_in_ego[1] > 0  # backward-left


def test_parse_fixed_metadata_rejects_unknown_and_duplicate_cameras():
    with pytest.raises(KeyError):
        parse_fixed_metadata(
            {"device_extrinsics": [_camera_entry("camera_hood", np.eye(3))]}
        )
    with pytest.raises(ValueError):
        parse_fixed_metadata(
            {
                "device_extrinsics": [
                    _camera_entry("camera_left", np.eye(3)),
                    _camera_entry("camera_left_repeater", np.eye(3)),
                ]
            }
        )


def test_folder_key_from_device_name():
    assert folder_key_from_device_name("camera_front") == "FRONT"
    assert folder_key_from_device_name("camera_left_repeater") == "LEFT_REPEATER"


def test_direction_map_covers_both_rig_layouts():
    directions_4cam = {
        DEVICE_NAME_TO_DIRECTION[f"camera_{name}"]
        for name in ("front", "rear", "left", "right")
    }
    directions_6cam = {
        DEVICE_NAME_TO_DIRECTION[f"camera_{name}"]
        for name in (
            "front",
            "rear",
            "left_repeater",
            "right_repeater",
            "left_pillar",
            "right_pillar",
        )
    }
    assert len(directions_4cam) == 4 and len(directions_6cam) == 6


# --------------------------------------------------------------------------- #
# Trip-insight helpers / timestamp localization
# --------------------------------------------------------------------------- #
def test_trip_insight_helpers_flat_and_wrapped_minutes():
    insight = {
        "timezone": "Europe/Zurich",
        "minutes": [{"country": "Switzerland", "region": "Vaud"}],
    }
    assert io.trip_timezone(insight) == "Europe/Zurich"
    assert io.trip_location(insight) == ("Switzerland", "Vaud")
    wrapped = {
        "minutes": [
            {"2025-11-20_05-34": {"country": "United States", "region": "Florida"}}
        ]
    }
    assert io.trip_location(wrapped) == ("United States", "Florida")
    assert io.trip_timezone({}) is None
    assert io.trip_location({}) == ("", "")


def test_localize_timestamps_zurich_matches_epoch():
    # 2025-11-20T05:34:02 Europe/Zurich (CET, UTC+1) == 1763613242 epoch --
    # the real value cross-checked against a trip's startEpochMs.
    series = pd.Series(pd.to_datetime(["2025-11-20T05:34:02.000"]))
    epoch = io.localize_timestamps(series, "Europe/Zurich")
    np.testing.assert_allclose(epoch, [1763613242.0], atol=1e-6)
    epoch_utc = io.localize_timestamps(series, None)
    np.testing.assert_allclose(epoch_utc, [1763613242.0 + 3600.0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Synthetic trip (hermetic end-to-end)
# --------------------------------------------------------------------------- #
_TZ = "Europe/Zurich"
_WIDTH, _HEIGHT = 64, 48
_FPS = 10.0


def _write_video(path: Path, n_frames: int) -> None:
    """Write an mp4 whose frame ``i`` is the solid gray level ``10 * i``."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    if not writer.isOpened():  # pragma: no cover - codec-less environments
        pytest.skip("cv2.VideoWriter cannot encode mp4v in this environment")
    for index in range(n_frames):
        frame = np.full((_HEIGHT, _WIDTH, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _frame_gray_level(image: np.ndarray) -> float:
    return float(image.mean())


def _write_camera_clip(
    clip_dir: Path,
    camera_key: str,
    start: datetime,
    n_frames: int,
    fps: float,
    fix_every: int,
    start_lat: float = 46.0,
    start_lon: float = 7.0,
    speed_mps: float = 11.12,
) -> None:
    """Write one camera folder: an encoded mp4 + a CSV with ``na`` gaps.

    GPS fixes land every ``fix_every``-th row and step the longitude east by
    1e-4 deg (~7.7 m at lat 46) per row interval; heading is due east.
    """
    folder = clip_dir / f"{camera_key}_FOLDER"
    folder.mkdir(parents=True)
    stamp = start.strftime("%Y-%m-%d_%H-%M-%S")
    _write_video(folder / f"{camera_key}_{stamp}.mp4", n_frames)
    rows = []
    for index in range(n_frames):
        row_time = start + timedelta(seconds=index / fps)
        timestamp = row_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        if index % fix_every == 0:
            longitude = start_lon + 1.0e-4 * (index / fix_every)
            rows.append(
                f"{timestamp},{index + 1},{start_lat},{longitude:.7f},5.0,"
                f"{speed_mps},0.0,{speed_mps},90.0,na,90.0"
            )
        else:
            rows.append(f"{timestamp},{index + 1},na,na,na,na,na,na,na,na,na")
    header = (
        "timestamp,frame_number,GPS_latitude_deg,GPS_longitude_deg,"
        "horizontal_accuracy_m,speed_mps,velocity_north_mps,velocity_east_mps,"
        "heading_deg,heading_accuracy_deg,image_direction"
    )
    (folder / f"{camera_key}_{stamp}.csv").write_text(
        "\n".join([header] + rows) + "\n", encoding="utf-8"
    )


def _write_synthetic_trip(root: Path) -> Path:
    """A 2-clip trip: FRONT (10 fps) + REAR (5 fps); clip 2 has no REAR."""
    trip_dir = root / "Testland" / "aaaa-bbbb_1"
    trip_dir.mkdir(parents=True)
    front_rotation = np.eye(3)
    rear_rotation = np.diag([-1.0, -1.0, 1.0])
    (trip_dir / "fixed_metadata.json").write_text(
        json.dumps(
            {
                "trip_identifier": "aaaa-bbbb",
                "reference_frame": "ground_nominal",
                "device_extrinsics": [
                    _camera_entry("camera_front", front_rotation, cx=32.0, cy=24.0),
                    _camera_entry(
                        "camera_rear", rear_rotation, tx=-88.0, cx=32.0, cy=24.0
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    (trip_dir / "trip_insight.json").write_text(
        json.dumps(
            {
                "timezone": _TZ,
                "minutes": [{"country": "Testland", "region": "Testregion"}],
            }
        ),
        encoding="utf-8",
    )
    clip1 = trip_dir / "10-00-00"
    start1 = datetime(2025, 6, 1, 10, 0, 0)
    _write_camera_clip(clip1, "FRONT", start1, n_frames=20, fps=_FPS, fix_every=2)
    _write_camera_clip(clip1, "REAR", start1, n_frames=10, fps=_FPS / 2, fix_every=2)
    clip2 = trip_dir / "10-00-02"
    start2 = datetime(2025, 6, 1, 10, 0, 2)
    _write_camera_clip(
        clip2, "FRONT", start2, n_frames=20, fps=_FPS, fix_every=2, start_lon=7.002
    )
    return trip_dir


@pytest.fixture(scope="module")
def synthetic_trip(tmp_path_factory):
    root = tmp_path_factory.mktemp("natix_synthetic")
    _write_synthetic_trip(root)
    return root


def test_discover_trips_and_clips_on_synthetic_layout(synthetic_trip):
    trips = io.discover_trips(str(synthetic_trip))
    assert [t.trip_name for t in trips] == ["aaaa-bbbb_1"]
    clips = io.discover_clips(trips[0])
    assert [c.clip_name for c in clips] == ["10-00-00", "10-00-02"]
    assert set(clips[0].cameras) == {"FRONT", "REAR"}
    assert set(clips[1].cameras) == {"FRONT"}
    with pytest.raises(FileNotFoundError):
        io.discover_trips(str(synthetic_trip / "does_not_exist"))


def test_synthetic_frame_table_alignment(synthetic_trip):
    trip = io.discover_trips(str(synthetic_trip))[0]
    clips = io.discover_clips(trip)
    table = io.build_trip_frame_table(clips, ["FRONT", "REAR"], _TZ)
    # 10 fixes per clip (every 2nd of 20 front rows), two clips.
    assert table.n_frames == 20
    assert bool(np.all(np.diff(table.timestamps) > 0))
    expected_start = datetime(
        2025, 6, 1, 10, 0, 0, tzinfo=zoneinfo.ZoneInfo(_TZ)
    ).timestamp()
    np.testing.assert_allclose(table.timestamps[0], expected_start, atol=1e-6)
    # Front frame numbers are the fix rows themselves (1-based: 1, 3, 5, ...).
    np.testing.assert_array_equal(
        table.frame_numbers["FRONT"][:10], np.arange(1, 20, 2)
    )
    # The 5 fps rear stream matches each fix to its nearest rear frame.
    np.testing.assert_array_equal(
        table.frame_numbers["REAR"][:4], np.array([1, 2, 3, 4])
    )
    # Clip 2 ships no rear camera at all.
    assert np.all(table.frame_numbers["REAR"][10:] == -1)
    assert np.all(table.clip_indices == np.repeat([0, 1], 10))


def test_synthetic_converter_iterates_expected_tasks(synthetic_trip, tmp_path):
    processor = NatixMulticamDatasetProcessor(
        common_output_path=str(tmp_path), split="all", context_aggregators=[]
    )
    converter = NatixMulticamDatasetConverter(
        source_processor=processor,
        input_path=str(synthetic_trip),
        split="all",
        do_parallel_processing=False,
        arguments={"frame_stride": 2},
    )
    tasks = list(converter._source_dataset_iterator)
    assert [index for _, index in tasks] == list(range(0, 20, 2))
    assert {ref.trip_name for ref, _ in tasks} == {"aaaa-bbbb_1"}


@pytest.fixture(scope="module")
def synthetic_built(synthetic_trip, tmp_path_factory):
    out = tmp_path_factory.mktemp("natix_out")
    processor = NatixMulticamDatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    trip = io.discover_trips(str(synthetic_trip))[0]
    yield processor, trip
    processor.cleanup()


def test_synthetic_frame_decodes_the_matched_video_frames(synthetic_built):
    processor, trip = synthetic_built
    frame = processor._prepare_standardized_frame_data((trip, 3))
    assert set(frame.cameras) == {CameraDirection.FRONT, CameraDirection.REAR}
    # Fix 3 anchors front row 7 (frame_number 7 -> video frame 6 -> level 60)
    # and the 5 fps rear stream's nearest frame 4 (video frame 3 -> level 30).
    assert np.isclose(
        _frame_gray_level(frame.cameras[CameraDirection.FRONT].image), 60, atol=4
    )
    assert np.isclose(
        _frame_gray_level(frame.cameras[CameraDirection.REAR].image), 30, atol=4
    )
    # Repeat read returns the cached frame; the earlier index reopens cleanly.
    again = processor._prepare_standardized_frame_data((trip, 3))
    np.testing.assert_array_equal(
        frame.cameras[CameraDirection.FRONT].image,
        again.cameras[CameraDirection.FRONT].image,
    )


def test_synthetic_frame_pose_speed_and_index_metadata(synthetic_built):
    processor, trip = synthetic_built
    frame = processor._prepare_standardized_frame_data((trip, 2))
    # Fixes step 1e-4 deg east (~7.7 m at lat 46) per fix; due-east heading.
    x = float(frame.global_position.get(TC.X)[0, 0])
    y = float(frame.global_position.get(TC.Y)[0, 0])
    heading = float(frame.global_position.get(TC.HEADING)[0, 0])
    speed = float(frame.global_position.get(TC.SPEED)[0, 0])
    assert np.isclose(x, 2 * 7.71, atol=0.2)
    assert np.isclose(y, 0.0, atol=0.05)
    assert np.isclose(heading, 0.0, atol=1e-6)  # east == FLU yaw 0
    assert np.isclose(speed, 11.12, atol=1e-6)
    pose = frame.aux_data["pose_matrix"]
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
    assert frame.extra_index_data == {
        "trip": "aaaa-bbbb_1",
        "country": "Testland",
        "region": "Testregion",
        "camera_count": 2,
    }
    # Second-clip frames carry only the front camera (no REAR folder there).
    frame_clip2 = processor._prepare_standardized_frame_data((trip, 15))
    assert set(frame_clip2.cameras) == {CameraDirection.FRONT}
    with pytest.raises(IndexError):
        processor._prepare_standardized_frame_data((trip, 999))


def test_build_trip_frame_table_drops_non_monotonic_timestamps(tmp_path):
    # Two clips whose CSV timelines overlap: the second clip starts *before*
    # the first ends, so its early frames must be dropped by the guard.
    trip_dir = tmp_path / "Testland" / "overlap_1"
    trip_dir.mkdir(parents=True)
    clip1 = trip_dir / "10-00-00"
    clip2 = trip_dir / "10-00-01"
    _write_camera_clip(clip1, "FRONT", datetime(2025, 6, 1, 10, 0, 0), 20, _FPS, 2)
    _write_camera_clip(clip2, "FRONT", datetime(2025, 6, 1, 10, 0, 1), 20, _FPS, 2)
    clips = [
        io.ClipRef(
            path=str(clip),
            clip_name=clip.name,
            start=start,
            cameras=io._camera_files_for_clip(clip),
        )
        for clip, start in (
            (clip1, datetime(2025, 6, 1, 10, 0, 0)),
            (clip2, datetime(2025, 6, 1, 10, 0, 1)),
        )
    ]
    table = io.build_trip_frame_table(clips, ["FRONT"], _TZ)
    assert table.n_frames < 20
    assert bool(np.all(np.diff(table.timestamps) > 0))


# --------------------------------------------------------------------------- #
# Real-data checks (skipped without data)
# --------------------------------------------------------------------------- #
def _first_real_trip():
    if not _NATIX_ROOT or not Path(_NATIX_ROOT).exists():
        return None
    try:
        trips = io.discover_trips(_NATIX_ROOT)
    except FileNotFoundError:
        return None
    return trips[0] if trips else None


@pytest.fixture(scope="module")
def real_built(tmp_path_factory):
    trip = _first_real_trip()
    if trip is None:
        pytest.skip("no NATIX data available (set NATIX_MULTICAM_ROOT)")
    out = tmp_path_factory.mktemp("natix_real")
    processor = NatixMulticamDatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    processor._refresh_trip_cache(trip)
    table = processor._table
    assert table is not None and table.n_frames > 0
    frame = processor._prepare_standardized_frame_data((trip, table.n_frames // 2))
    yield processor, trip, table, frame
    processor.cleanup()


def test_real_frame_cameras_match_their_calibration(real_built):
    processor, _trip, _table, frame = real_built
    assert frame.cameras, "mid-trip frame should carry at least one camera"
    assert set(frame.cameras) <= set(processor._calibrations)
    for direction, camera in frame.cameras.items():
        assert camera.image.ndim == 3 and camera.image.dtype == np.uint8
        height, width = camera.image.shape[:2]
        # Principal point near the actual footage centre -- guards both the
        # intrinsics parse and the native-size front-camera contract.
        assert abs(camera.intrinsics[0, 2] - width / 2) < 0.3 * width, direction
        assert abs(camera.intrinsics[1, 2] - height / 2) < 0.3 * height, direction
        rotation = camera.extrinsics[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)


def test_real_first_fix_matches_trip_insight_epoch(real_built):
    _processor, trip, table, _frame = real_built
    start_epoch_ms = io.read_trip_insight(trip).get("startEpochMs")
    if start_epoch_ms is None:
        pytest.skip("trip_insight.json carries no startEpochMs")
    # A wrong timezone would be off by whole hours; the data itself is only
    # aligned to within a few seconds.
    assert abs(table.timestamps[0] - start_epoch_ms / 1000.0) < 30.0


def test_real_heading_matches_gps_displacement(real_built):
    processor, _trip, table, _frame = real_built
    poses = processor._poses_world_from_ego
    xy = poses[:, :2, 3]
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    deltas = np.diff(xy, axis=0)
    moving = np.hypot(deltas[:, 0], deltas[:, 1]) > 1.0
    if moving.sum() < 10:
        pytest.skip("trip has too little motion to check the heading")
    displacement_yaw = np.arctan2(deltas[:, 1], deltas[:, 0])[moving]
    error = displacement_yaw - yaw[1:][moving]
    error = np.degrees(np.abs(np.arctan2(np.sin(error), np.cos(error))))
    assert np.median(error) < 20.0


def test_real_pose_is_rigid_and_finite(real_built):
    _processor, _trip, _table, frame = real_built
    pose = frame.aux_data["pose_matrix"]
    assert pose.shape == (4, 4) and np.isfinite(pose).all()
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-9)


def test_real_forward_only_reader_is_deterministic(real_built):
    processor, trip, table, _frame = real_built
    index = table.n_frames // 2
    first = processor._prepare_standardized_frame_data((trip, index))
    second = processor._prepare_standardized_frame_data((trip, index))
    for direction in first.cameras:
        np.testing.assert_array_equal(
            first.cameras[direction].image, second.cameras[direction].image
        )
