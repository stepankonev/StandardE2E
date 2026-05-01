"""Tests for ``HDMapEgoCropAggregator`` (PR 2 step 2.4b).

Covers:
- Concrete subclass end-to-end: parses a hand-built ``RawSegmentHDMap``,
  crops per frame using each frame's ``aux_data["pose_matrix"]``,
  writes per-frame ego ``HDMapData`` into the same ``.npz`` archive.
- Idempotency: running the aggregator twice produces byte-equal cache
  files (catches accumulator / mutation bugs).
- ``FrameLoader``-style end-to-end load: reading the per-frame ``.npz``
  back yields ``HDMapData`` (the ego type), with no HD-map-specific
  load-time code path.
- Memory cleanup: after ``_process_segment`` returns, the instance does
  not retain a reference to the parsed segment map.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from standard_e2e.caching.segment_context.hd_map_ego_crop import (
    HDMapEgoCropAggregator,
)
from standard_e2e.data_structures import (
    HDMapData,
    Lane,
    RawSegmentHDMap,
    TransformedFrameData,
)
from standard_e2e.enums import LaneType, Modality


def _ego_pose_at(x: float, y: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    pose[1, 3] = y
    return pose


class _FakeAggregator(HDMapEgoCropAggregator):
    """Concrete subclass that returns a hand-built segment map."""

    def __init__(self, data_path: str, raw_map: RawSegmentHDMap, **kwargs):
        super().__init__(
            data_path=data_path, source_data_path="<unused>", x_range=50.0, y_range=50.0,
            **kwargs,
        )
        self._fixture_map = raw_map
        self.parse_calls = 0

    def _parse_world_segment_map(self, segment_id: str) -> RawSegmentHDMap:
        self.parse_calls += 1
        return self._fixture_map


def _write_frames(tmp_path: Path, segment_id: str, ego_xs: list[float]) -> list[TransformedFrameData]:
    frames: list[TransformedFrameData] = []
    for i, x in enumerate(ego_xs):
        frame = TransformedFrameData(
            dataset_name="ds",
            split="train",
            segment_id=segment_id,
            frame_id=i,
            timestamp=float(i),
            aux_data={"pose_matrix": _ego_pose_at(x, 0.0)},
        )
        out_path = tmp_path / frame.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_npz(str(out_path))
        frames.append(frame)
    return frames


def _index_df(frames: list[TransformedFrameData]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )


def _segment_map_with_two_lanes() -> RawSegmentHDMap:
    """Two lanes at world x = 0 and world x = 1000."""
    return RawSegmentHDMap(
        lanes=[
            Lane(
                centerline=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
                lane_type=LaneType.VEHICLE,
            ),
            Lane(
                centerline=np.array([[1000, 0, 0], [1010, 0, 0]], dtype=np.float32),
                lane_type=LaneType.VEHICLE,
            ),
        ]
    )


def test_aggregator_writes_per_frame_ego_hd_map(tmp_path: Path):
    frames = _write_frames(tmp_path, "segH", [0.0, 1000.0])
    aggr = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr.process(_index_df(frames))

    # Frame 0 (ego at world x=0) should see only the lane near origin.
    f0 = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[0].filename))
    payload0 = f0.get_modality_data(Modality.HD_MAP)
    assert isinstance(payload0, HDMapData)
    assert len(payload0.lanes) == 1
    np.testing.assert_allclose(
        payload0.lanes[0].centerline,
        np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        atol=1e-5,
    )

    # Frame 1 (ego at world x=1000) should see only the far lane (now at ego origin).
    f1 = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[1].filename))
    payload1 = f1.get_modality_data(Modality.HD_MAP)
    assert isinstance(payload1, HDMapData)
    assert len(payload1.lanes) == 1
    np.testing.assert_allclose(
        payload1.lanes[0].centerline,
        np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        atol=1e-5,
    )


def test_aggregator_parses_segment_map_once_per_segment(tmp_path: Path):
    frames = _write_frames(tmp_path, "segH", [0.0, 1.0, 2.0])
    aggr = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr.process(_index_df(frames))
    assert aggr.parse_calls == 1


def test_aggregator_is_idempotent(tmp_path: Path):
    frames = _write_frames(tmp_path, "segH", [0.0, 5.0])
    aggr = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr.process(_index_df(frames))

    first_run_bytes = []
    for f in frames:
        with open(tmp_path / f.filename, "rb") as fp:
            first_run_bytes.append(fp.read())

    aggr2 = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr2.process(_index_df(frames))

    for f, prev in zip(frames, first_run_bytes):
        with open(tmp_path / f.filename, "rb") as fp:
            assert fp.read() == prev


def test_aggregator_releases_segment_map_after_process(tmp_path: Path):
    """No leftover ``RawSegmentHDMap`` reference on the instance after
    a segment finishes (regression guard against accidental caching of
    segment maps that would balloon memory in a multi-segment run)."""
    frames = _write_frames(tmp_path, "segH", [0.0, 1.0])
    aggr = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr.process(_index_df(frames))

    # Walk the instance dict; confirm no RawSegmentHDMap value lingers
    # outside the deliberate fixture-holder we put there.
    for attr_name, value in vars(aggr).items():
        if attr_name == "_fixture_map":
            continue
        assert not isinstance(value, RawSegmentHDMap), (
            f"aggregator retained a RawSegmentHDMap on {attr_name!r}"
        )


def test_frame_loader_returns_ego_hd_map_with_no_special_path(tmp_path: Path):
    """End-to-end load via plain TransformedFrameData.from_npz returns
    HDMapData (no HD-map-specific code path needed)."""
    frames = _write_frames(tmp_path, "segH", [0.0])
    aggr = _FakeAggregator(str(tmp_path), _segment_map_with_two_lanes())
    aggr.process(_index_df(frames))

    reloaded = TransformedFrameData.from_npz(
        os.path.join(tmp_path, frames[0].filename),
        required_modalities=[Modality.HD_MAP],
    )
    payload = reloaded.get_modality_data(Modality.HD_MAP)
    assert isinstance(payload, HDMapData)
    assert not isinstance(payload, RawSegmentHDMap)
