"""End-to-end test for ``WaymoHDMapEgoCropAggregator`` (PR 2 step 2.5).

Builds a synthetic Waymo tfrecord (1 segment, 3 frames, hand-known
``map_features``) on disk, runs the aggregator, and asserts each
frame's per-frame ``Modality.HD_MAP`` payload matches the hand-computed
ego crop. Also verifies that no segment-wide artifact lingers on the
aggregator after processing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

# pylint: disable=no-name-in-module
from standard_e2e.caching.src_datasets.waymo_perception.hd_map_ego_crop import (
    WaymoHDMapEgoCropAggregator,
)
from standard_e2e.data_structures import (
    HDMapData,
    RawSegmentHDMap,
    TransformedFrameData,
)
from standard_e2e.enums import Modality
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame
from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import LaneCenter

pytestmark = pytest.mark.waymo


def _add_polyline(repeated_points, pts: list[tuple[float, float, float]]):
    for x, y, z in pts:
        p = repeated_points.add()
        p.x = x
        p.y = y
        p.z = z


def _build_segment_tfrecord(tfrecord_dir: Path, segment_id: str) -> Path:
    """Write a tfrecord with one frame whose map_features contain a
    single lane at world (offset_x + 0..10, 0, 0) (offset = 100, 200, 5)."""
    tfrecord_dir.mkdir(parents=True, exist_ok=True)
    path = tfrecord_dir / f"{segment_id}.tfrecord"

    frame = Frame()
    frame.context.name = segment_id
    frame.timestamp_micros = 0
    frame.map_pose_offset.x = 100.0
    frame.map_pose_offset.y = 200.0
    frame.map_pose_offset.z = 5.0
    feat = frame.map_features.add()
    feat.id = 1
    feat.lane.type = LaneCenter.TYPE_SURFACE_STREET
    _add_polyline(feat.lane.polyline, [(0, 0, 0), (5, 0, 0), (10, 0, 0)])

    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(frame.SerializeToString())
    return path


def _ego_pose_world(x: float, y: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    pose[1, 3] = y
    return pose


def _write_per_frame_npz(
    cache_path: Path, segment_id: str, ego_world_xs: list[float]
) -> list[TransformedFrameData]:
    frames: list[TransformedFrameData] = []
    for i, x in enumerate(ego_world_xs):
        f = TransformedFrameData(
            dataset_name="waymo_perception",
            split="validation",
            segment_id=segment_id,
            frame_id=i,
            timestamp=float(i),
            aux_data={"pose_matrix": _ego_pose_world(x, 200.0)},
        )
        out = cache_path / f.filename
        out.parent.mkdir(parents=True, exist_ok=True)
        f.to_npz(str(out))
        frames.append(f)
    return frames


def _index_df(frames: list[TransformedFrameData]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )


@pytest.fixture
def synthetic_waymo_segment(tmp_path: Path):
    segment_id = "segment-synthetic-12345-with-camera-labels"
    source_path = tmp_path / "source"
    cache_path = tmp_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    _build_segment_tfrecord(source_path, segment_id)
    return segment_id, source_path, cache_path


def test_waymo_hd_map_aggregator_produces_ego_hd_map_per_frame(synthetic_waymo_segment):
    segment_id, source_path, cache_path = synthetic_waymo_segment
    # Ego at world (105, 200, ?): the lane in world frame goes from
    # (100, 200, 5) to (110, 200, 5). After ego crop with ego at
    # (105, 200, 0), the centerline in ego frame is (-5, 0, 5) -> (5, 0, 5).
    frames = _write_per_frame_npz(cache_path, segment_id, [105.0, 105.0, 105.0])

    aggr = WaymoHDMapEgoCropAggregator(
        data_path=str(cache_path),
        source_data_path=str(source_path),
        x_range=50.0,
        y_range=50.0,
    )
    aggr.process(_index_df(frames))

    for f in frames:
        reloaded = TransformedFrameData.from_npz(os.path.join(cache_path, f.filename))
        payload = reloaded.get_modality_data(Modality.HD_MAP)
        assert isinstance(payload, HDMapData)
        assert len(payload.lanes) == 1
        # World centerline: (100, 200, 5) -> (105, 200, 5) -> (110, 200, 5).
        # Ego at (105, 200, 0) -> ego centerline: (-5, 0, 5), (0, 0, 5), (5, 0, 5).
        expected = np.array([[-5, 0, 5], [0, 0, 5], [5, 0, 5]], dtype=np.float32)
        np.testing.assert_allclose(payload.lanes[0].centerline, expected, atol=1e-4)


def test_waymo_hd_map_aggregator_no_segment_map_state_after_run(
    synthetic_waymo_segment,
):
    segment_id, source_path, cache_path = synthetic_waymo_segment
    frames = _write_per_frame_npz(cache_path, segment_id, [105.0])

    aggr = WaymoHDMapEgoCropAggregator(
        data_path=str(cache_path),
        source_data_path=str(source_path),
        x_range=50.0,
        y_range=50.0,
    )
    aggr.process(_index_df(frames))

    for attr_name, value in vars(aggr).items():
        assert not isinstance(
            value, RawSegmentHDMap
        ), f"aggregator retained RawSegmentHDMap on {attr_name!r}"


def test_waymo_hd_map_aggregator_raises_when_segment_missing(tmp_path: Path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "source"
    source_path.mkdir(parents=True, exist_ok=True)

    frames = _write_per_frame_npz(cache_path, "segment-missing", [0.0])
    aggr = WaymoHDMapEgoCropAggregator(
        data_path=str(cache_path),
        source_data_path=str(source_path),
        x_range=50.0,
        y_range=50.0,
    )
    with pytest.raises(FileNotFoundError):
        aggr.process(_index_df(frames))
