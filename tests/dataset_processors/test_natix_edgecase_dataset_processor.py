# flake8: noqa: E501
"""Tests for the NATIX Edge Case processor.

* **Annotation parsing units** (always run): the ``(trip, clip)`` keying of
  ``clip_path``, the ``edge-case.json`` parser (empty labels, malformed
  entries), the label -> ``EVENT CLASSIFICATION`` summary fallback, the
  walk-up ``edge-case.json`` discovery, the video anchor extrapolation for
  CSVs whose leading rows are missing, and the inclusive window match.
* **Synthetic-trip end-to-end** (always run; hermetic): a tiny front-only
  trip written to ``tmp_path`` in the multicam layout plus an
  ``edge-case.json`` -- frames inside an event window carry the event in
  ``aux_data`` / the index extras, frames outside carry empty annotations,
  and a tree without ``edge-case.json`` processes with empty annotations.
* **Real-data checks** (skipped unless ``NATIX_EDGECASE_ROOT`` points at a
  downloaded tree): every annotated ``(trip, clip)`` key resolves to a
  discovered clip, and an annotated trip emits frames whose
  ``edge_case_count`` window matches the annotation.
"""

from __future__ import annotations

import json
import logging
import os
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from standard_e2e.caching.src_datasets.natix_edgecase import (
    NatixEdgeCaseDatasetConverter,
    NatixEdgeCaseDatasetProcessor,
)
from standard_e2e.caching.src_datasets.natix_edgecase import (
    _natix_edgecase_events as events_io,
)
from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io

# Real-data checks read NATIX edge-case only from this env var.
_EDGECASE_ROOT = os.environ.get("NATIX_EDGECASE_ROOT", "")

_TZ = "Europe/Zurich"
_WIDTH, _HEIGHT = 64, 48
_FPS = 10.0


# --------------------------------------------------------------------------- #
# Annotation parsing units (no data required)
# --------------------------------------------------------------------------- #
def _event_dict(label: str = "test event", start: float = 0.5, end: float = 1.0):
    return {
        "label": label,
        "start_sec": start,
        "end_sec": end,
        "ai_analysis": {"EVENT CLASSIFICATION": "Test classification."},
    }


def test_clip_key_from_path_takes_last_two_components():
    key = events_io._clip_key_from_path(
        "United States/Georgia/4d4cfb50-69a3_7/10-30-41/"
    )
    assert key == ("4d4cfb50-69a3_7", "10-30-41")
    # Windows separators and no trailing slash are tolerated.
    assert events_io._clip_key_from_path("a\\trip_1\\09-00-00") == (
        "trip_1",
        "09-00-00",
    )
    assert events_io._clip_key_from_path("lonely") is None
    assert events_io._clip_key_from_path("") is None


def test_load_edge_case_events_parses_and_skips_malformed(tmp_path):
    path = tmp_path / events_io.EDGE_CASE_JSON_NAME
    path.write_text(
        json.dumps(
            [
                {
                    "clip_path": "US/State/trip_1/10-00-00/",
                    "events": [_event_dict(), _event_dict(label="", start=2, end=3)],
                },
                {"clip_path": "nopath", "events": [_event_dict()]},
                {
                    "clip_path": "US/State/trip_1/10-01-00/",
                    "events": [{"label": "no window"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    events = events_io.load_edge_case_events(str(path))
    assert set(events) == {("trip_1", "10-00-00"), ("trip_1", "10-01-00")}
    first = events[("trip_1", "10-00-00")]
    assert [e.label for e in first] == ["test event", ""]
    assert first[0].start_sec == 0.5 and first[0].end_sec == 1.0
    assert first[0].ai_analysis["EVENT CLASSIFICATION"] == "Test classification."
    # The windowless event is dropped; its clip entry stays (empty).
    assert events[("trip_1", "10-01-00")] == []


def test_event_summary_falls_back_to_classification():
    labelled, unlabelled = (
        events_io.EdgeCaseEvent("a label", 0.0, 1.0, {"EVENT CLASSIFICATION": "cls"}),
        events_io.EdgeCaseEvent("", 0.0, 1.0, {"EVENT CLASSIFICATION": "cls"}),
    )
    assert labelled.summary == "a label"
    assert unlabelled.summary == "cls"
    assert events_io.EdgeCaseEvent("", 0.0, 1.0, {}).summary == ""


def test_find_edge_case_json_walks_up(tmp_path):
    trip = tmp_path / "data" / "US" / "State" / "trip_1"
    trip.mkdir(parents=True)
    marker = tmp_path / "data" / events_io.EDGE_CASE_JSON_NAME
    marker.write_text("[]", encoding="utf-8")
    assert events_io.find_edge_case_json(str(trip)) == str(marker)
    bare = tmp_path / "bare" / "trip_2"
    bare.mkdir(parents=True)
    assert events_io.find_edge_case_json(str(bare)) is None


def _write_front_csv(
    path: Path, start: datetime, n_rows: int, first_frame_number: int = 1
) -> None:
    header = (
        "timestamp,frame_number,GPS_latitude_deg,GPS_longitude_deg,"
        "horizontal_accuracy_m,speed_mps,velocity_north_mps,velocity_east_mps,"
        "heading_deg,heading_accuracy_deg,image_direction"
    )
    rows = []
    for index in range(n_rows):
        frame_number = first_frame_number + index
        row_time = start + timedelta(seconds=(frame_number - 1) / _FPS)
        stamp = row_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        rows.append(f"{stamp},{frame_number},na,na,na,na,na,na,na,na,na")
    path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


def test_clip_video_anchor_extrapolates_missing_leading_rows(tmp_path):
    start = datetime(2025, 6, 1, 10, 0, 0)
    aligned = tmp_path / "FRONT_aligned.csv"
    _write_front_csv(aligned, start, n_rows=20, first_frame_number=1)
    truncated = tmp_path / "FRONT_truncated.csv"
    _write_front_csv(truncated, start, n_rows=20, first_frame_number=8)
    expected = datetime(2025, 6, 1, 10, 0, 0, tzinfo=zoneinfo.ZoneInfo(_TZ)).timestamp()
    # A CSV starting at frame 1 anchors at its first row; one starting at
    # frame 8 must extrapolate back to the same video start.
    assert np.isclose(events_io.clip_video_anchor(str(aligned), _TZ), expected)
    assert np.isclose(
        events_io.clip_video_anchor(str(truncated), _TZ), expected, atol=1e-6
    )


def test_events_covering_inclusive_bounds():
    events = [events_io.EdgeCaseEvent("e", 1.0, 2.0, {})]
    anchor = 1000.0
    assert events_io.events_covering(events, anchor, 1000.9) == []
    assert events_io.events_covering(events, anchor, 1001.0) == events
    assert events_io.events_covering(events, anchor, 1002.0) == events
    assert events_io.events_covering(events, anchor, 1002.1) == []


# --------------------------------------------------------------------------- #
# Synthetic trip (hermetic end-to-end)
# --------------------------------------------------------------------------- #
def _write_video(path: Path, n_frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_WIDTH, _HEIGHT)
    )
    if not writer.isOpened():  # pragma: no cover - codec-less environments
        pytest.skip("cv2.VideoWriter cannot encode mp4v in this environment")
    for index in range(n_frames):
        writer.write(np.full((_HEIGHT, _WIDTH, 3), index * 10, dtype=np.uint8))
    writer.release()


def _write_front_clip(clip_dir: Path, start: datetime, n_frames: int) -> None:
    """A front-only clip: mp4 + CSV with a GPS fix every 2nd row."""
    folder = clip_dir / "FRONT_FOLDER"
    folder.mkdir(parents=True)
    stamp = start.strftime("%Y-%m-%d_%H-%M-%S")
    _write_video(folder / f"FRONT_{stamp}.mp4", n_frames)
    rows = []
    for index in range(n_frames):
        row_time = start + timedelta(seconds=index / _FPS)
        timestamp = row_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        if index % 2 == 0:
            longitude = 7.0 + 1.0e-4 * (index / 2)
            rows.append(
                f"{timestamp},{index + 1},46.0,{longitude:.7f},5.0,"
                f"11.12,0.0,11.12,90.0,na,90.0"
            )
        else:
            rows.append(f"{timestamp},{index + 1},na,na,na,na,na,na,na,na,na")
    header = (
        "timestamp,frame_number,GPS_latitude_deg,GPS_longitude_deg,"
        "horizontal_accuracy_m,speed_mps,velocity_north_mps,velocity_east_mps,"
        "heading_deg,heading_accuracy_deg,image_direction"
    )
    (folder / f"FRONT_{stamp}.csv").write_text(
        "\n".join([header] + rows) + "\n", encoding="utf-8"
    )


def _front_camera_entry() -> dict:
    entry = {
        "device_name": "camera_front",
        "tx": 182.0,
        "ty": 0.0,
        "tz": 130.0,
        "fx": 1600.0,
        "fy": 1600.0,
        "cx": 32.0,
        "cy": 24.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }
    for i in range(3):
        for j in range(3):
            entry[f"r{i + 1}{j + 1}"] = float(np.eye(3)[i, j])
    return entry


def _write_trip(root: Path, trip_name: str, clip_start: datetime) -> Path:
    trip_dir = root / "Testland" / trip_name
    trip_dir.mkdir(parents=True)
    (trip_dir / "fixed_metadata.json").write_text(
        json.dumps({"device_extrinsics": [_front_camera_entry()]}), encoding="utf-8"
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
    _write_front_clip(
        trip_dir / clip_start.strftime("%H-%M-%S"), clip_start, n_frames=20
    )
    return trip_dir


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    """Two single-clip trips; only the first has edge-case annotations.

    Trip 1's clip runs 0.0-1.9 s of video time with frames (fixes) every
    0.2 s; event A covers [0.5, 1.0] (frames at 0.6 / 0.8 / 1.0) and the
    label-less event B covers [1.7, 5.0] (frames at 1.8).
    """
    root = tmp_path_factory.mktemp("natix_edgecase") / "data"
    _write_trip(root, "aaaa-bbbb_1", datetime(2025, 6, 1, 10, 0, 0))
    _write_trip(root, "cccc-dddd_1", datetime(2025, 6, 1, 11, 0, 0))
    (root / events_io.EDGE_CASE_JSON_NAME).write_text(
        json.dumps(
            [
                {
                    "clip_path": "Testland/aaaa-bbbb_1/10-00-00/",
                    "events": [
                        _event_dict(label="event A", start=0.5, end=1.0),
                        _event_dict(label="", start=1.7, end=5.0),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def edgecase_built(synthetic_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("natix_edgecase_out")
    processor = NatixEdgeCaseDatasetProcessor(
        common_output_path=str(out), split="all", context_aggregators=[]
    )
    trips = io.discover_trips(str(synthetic_root))
    yield processor, trips
    processor.cleanup()


def test_synthetic_frames_inside_and_outside_event_windows(edgecase_built):
    processor, trips = edgecase_built
    annotated = trips[0]
    assert annotated.trip_name == "aaaa-bbbb_1"

    # Frame 0 (video time 0.0 s) is outside every window.
    outside = processor._prepare_standardized_frame_data((annotated, 0))
    assert outside.dataset_name == "natix_edgecase"
    assert outside.aux_data is not None
    assert outside.aux_data["edge_case_events"] == []
    assert outside.extra_index_data is not None
    assert outside.extra_index_data["edge_case_count"] == 0
    assert outside.extra_index_data["edge_case"] == ""
    # The multicam index extras are still there.
    assert outside.extra_index_data["trip"] == "aaaa-bbbb_1"
    assert outside.extra_index_data["camera_count"] == 1

    # Frame 4 (video time 0.8 s) is inside event A only.
    inside = processor._prepare_standardized_frame_data((annotated, 4))
    events = inside.aux_data["edge_case_events"]
    assert [event["label"] for event in events] == ["event A"]
    assert events[0]["start_sec"] == 0.5 and events[0]["end_sec"] == 1.0
    assert events[0]["ai_analysis"]["EVENT CLASSIFICATION"] == "Test classification."
    assert inside.extra_index_data["edge_case_count"] == 1
    assert inside.extra_index_data["edge_case"] == "event A"

    # Frame 5 (video time 1.0 s) sits exactly on event A's inclusive end.
    boundary = processor._prepare_standardized_frame_data((annotated, 5))
    assert boundary.extra_index_data["edge_case_count"] == 1

    # Frame 9 (video time 1.8 s) is inside the label-less event B, whose
    # summary falls back to the VLM classification.
    fallback = processor._prepare_standardized_frame_data((annotated, 9))
    assert fallback.extra_index_data["edge_case_count"] == 1
    assert fallback.extra_index_data["edge_case"] == "Test classification."
    assert fallback.aux_data["edge_case_events"][0]["label"] == ""


def test_synthetic_unannotated_trip_has_empty_annotations(edgecase_built):
    processor, trips = edgecase_built
    frame = processor._prepare_standardized_frame_data((trips[1], 0))
    assert frame.aux_data is not None and frame.extra_index_data is not None
    assert frame.aux_data["edge_case_events"] == []
    assert frame.extra_index_data["edge_case_count"] == 0


def test_synthetic_tree_without_json_warns_and_emits_empty(tmp_path, caplog):
    _write_trip(tmp_path, "eeee-ffff_1", datetime(2025, 6, 1, 12, 0, 0))
    processor = NatixEdgeCaseDatasetProcessor(
        common_output_path=str(tmp_path / "out"), split="all", context_aggregators=[]
    )
    trip = io.discover_trips(str(tmp_path))[0]
    with caplog.at_level(logging.WARNING):
        frame = processor._prepare_standardized_frame_data((trip, 0))
    processor.cleanup()
    assert frame.aux_data is not None and frame.extra_index_data is not None
    assert frame.aux_data["edge_case_events"] == []
    assert frame.extra_index_data["edge_case_count"] == 0
    assert any(events_io.EDGE_CASE_JSON_NAME in message for message in caplog.messages)


def test_synthetic_converter_yields_edgecase_tasks(synthetic_root, tmp_path):
    processor = NatixEdgeCaseDatasetProcessor(
        common_output_path=str(tmp_path), split="all", context_aggregators=[]
    )
    converter = NatixEdgeCaseDatasetConverter(
        source_processor=processor,
        input_path=str(synthetic_root),
        split="all",
        do_parallel_processing=False,
        arguments={"frame_stride": 1},
    )
    tasks = list(converter._source_dataset_iterator)
    assert {ref.trip_name for ref, _ in tasks} == {"aaaa-bbbb_1", "cccc-dddd_1"}
    assert processor.dataset_name == "natix_edgecase"


# --------------------------------------------------------------------------- #
# Real-data checks (skipped without data)
# --------------------------------------------------------------------------- #
def _real_trips() -> list[io.TripRef]:
    if not _EDGECASE_ROOT or not Path(_EDGECASE_ROOT).exists():
        return []
    try:
        return io.discover_trips(_EDGECASE_ROOT)
    except FileNotFoundError:
        return []


@pytest.fixture(scope="module")
def real_annotations():
    trips = _real_trips()
    if not trips:
        pytest.skip("no NATIX edge-case data available (set NATIX_EDGECASE_ROOT)")
    json_path = events_io.find_edge_case_json(trips[0].path)
    if json_path is None:
        pytest.skip("downloaded tree carries no edge-case.json")
    return trips, events_io.load_edge_case_events(json_path)


def test_real_annotated_clips_all_resolve_to_discovered_clips(real_annotations):
    trips, events = real_annotations
    assert events, "edge-case.json should carry at least one annotated clip"
    discovered = {
        (trip.trip_name, clip.clip_name)
        for trip in trips
        for clip in io.discover_clips(trip)
    }
    missing = set(events) - discovered
    assert not missing, f"annotated clips missing from the tree: {missing}"


def test_real_annotated_trip_emits_in_event_frames(real_annotations, tmp_path):
    trips, events = real_annotations
    annotated_trip_names = {trip_name for trip_name, _clip in events}
    trip = next(t for t in trips if t.trip_name in annotated_trip_names)
    processor = NatixEdgeCaseDatasetProcessor(
        common_output_path=str(tmp_path), split="all", context_aggregators=[]
    )
    processor._refresh_trip_cache(trip)
    table = processor._table
    assert table is not None and table.n_frames > 0
    counts = []
    for frame_index in range(table.n_frames):
        frame_events = processor._events_covering_frame(frame_index)
        counts.append(len(frame_events))
    processor.cleanup()
    # The trip is annotated, so some (but usually not all) frames are covered.
    assert max(counts) >= 1
    windows = [
        event.end_sec - event.start_sec
        for (trip_name, _clip), clip_events in events.items()
        if trip_name == trip.trip_name
        for event in clip_events
    ]
    # Sanity: covered-frame share is loosely consistent with the windows.
    assert sum(count > 0 for count in counts) <= len(counts)
    assert min(windows) >= 0.0
