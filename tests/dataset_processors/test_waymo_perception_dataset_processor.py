"""Tests for WaymoPerceptionDatasetProcessor (construction + defaults)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.caching.adapters import (
    Detections3DIdentityAdapter,
    LidarPCIdentityAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import FutureDetectionsAggregator
from standard_e2e.caching.src_datasets.waymo_perception import (
    waymo_perception_dataset_processor as _wpdp,
)
from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    StandardFrameData,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import DetectionType, Modality
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame

pytestmark = pytest.mark.waymo


class _FakeRaw:
    """Minimal stand-in for the ``tf.Tensor`` that the processor receives.

    The processor only needs ``raw_frame_data.numpy()`` to return the
    serialized bytes of a Waymo ``Frame`` proto.
    """

    def __init__(self, payload: bytes):
        self._payload = payload

    def numpy(self) -> bytes:
        return self._payload


def test_waymo_perception_defaults(tmp_path: Path):
    proc = _wpdp.WaymoPerceptionDatasetProcessor(str(tmp_path), split="training")
    assert proc.dataset_name == "waymo_perception"
    assert set(["training", "validation", "testing"]).issubset(set(proc.allowed_splits))
    adapters = getattr(proc, "_adapters")  # Access internal for test
    adapter_types = {type(a) for a in adapters}
    assert PanoImageAdapter in adapter_types
    assert Detections3DIdentityAdapter in adapter_types
    assert LidarPCIdentityAdapter in adapter_types


def test_waymo_perception_hd_map_wired_when_source_data_path_set(tmp_path: Path):
    """When ``source_data_path`` is provided, the HD-map adapter and
    ``WaymoHDMapEgoCropAggregator`` are added to the defaults."""
    from standard_e2e.caching.adapters import HDMapIdentityAdapter
    from standard_e2e.caching.src_datasets.waymo_perception.hd_map_ego_crop import (
        WaymoHDMapEgoCropAggregator,
    )

    proc = _wpdp.WaymoPerceptionDatasetProcessor(
        str(tmp_path), split="training", source_data_path=str(tmp_path / "src")
    )
    adapter_types = {type(a) for a in getattr(proc, "_adapters")}
    assert HDMapIdentityAdapter in adapter_types
    aggregator_types = {type(a) for a in proc.context_aggregators}
    assert WaymoHDMapEgoCropAggregator in aggregator_types


def _build_standard_frame(
    segment_id: str,
    frame_id: int,
    timestamp: float,
    agent_x: float,
) -> StandardFrameData:
    """Build a synthetic StandardFrameData with a single detection at (agent_x, 0)."""
    detection = Detection3D(
        unique_agent_id="agent-0",
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            {
                TC.TIMESTAMP: [timestamp],
                TC.X: [agent_x],
                TC.Y: [0.0],
                TC.Z: [0.0],
                TC.HEADING: [0.0],
                TC.LENGTH: [4.0],
                TC.WIDTH: [2.0],
                TC.HEIGHT: [1.5],
            }
        ),
    )
    pose = np.eye(4, dtype=np.float32)
    return StandardFrameData(
        dataset_name="waymo_perception",
        split="training",
        segment_id=segment_id,
        frame_id=frame_id,
        timestamp=timestamp,
        global_position=Trajectory(
            {
                TC.TIMESTAMP: [timestamp],
                TC.X: [0.0],
                TC.Y: [0.0],
                TC.Z: [0.0],
                TC.HEADING: [0.0],
            }
        ),
        frame_detections_3d=FrameDetections3D(detections=[detection]),
        aux_data={"pose_matrix": pose},
    )


def _build_minimal_waymo_frame_proto(
    *, with_ghost: bool, with_real: bool
) -> WaymoFrame:
    """Build a Waymo ``Frame`` proto with the minimum fields the
    processor reads. Cameras/lasers are intentionally omitted so the
    test stays decoupled from the image / lidar code paths.
    """
    frame = WaymoFrame()
    frame.context.name = "ghost-test-segment"
    frame.timestamp_micros = 1_000_000  # 1 second since epoch
    # 4x4 identity pose, row-major (16 floats).
    frame.pose.transform.extend(
        [1.0 if i in (0, 5, 10, 15) else 0.0 for i in range(16)]
    )

    if with_ghost:
        ghost = frame.laser_labels.add()
        ghost.id = "ghost-agent"
        ghost.type = 1  # VEHICLE
        ghost.box.center_x = 0.0
        ghost.box.center_y = 0.0
        ghost.box.center_z = 0.0
        ghost.box.heading = 0.0
        ghost.box.length = 4.0
        ghost.box.width = 2.0
        ghost.box.height = 1.5

    if with_real:
        real = frame.laser_labels.add()
        real.id = "real-agent"
        real.type = 1  # VEHICLE
        real.box.center_x = 12.0
        real.box.center_y = 3.0
        real.box.center_z = 0.5
        real.box.heading = 0.0
        real.box.length = 4.5
        real.box.width = 2.1
        real.box.height = 1.6

    return frame


def test_ghost_label_at_origin_is_skipped(tmp_path: Path, caplog):
    """A laser_label with center (0,0,0) is treated as malformed: the
    processor warns and skips it instead of emitting a ghost detection
    at the ego origin into the cache."""
    proc = _wpdp.WaymoPerceptionDatasetProcessor(str(tmp_path), split="training")
    proto = _build_minimal_waymo_frame_proto(with_ghost=True, with_real=True)
    raw = _FakeRaw(proto.SerializeToString())

    with caplog.at_level(logging.WARNING, logger=_wpdp.__name__):
        std = proc._prepare_standardized_frame_data(raw)  # noqa: SLF001

    assert isinstance(std, StandardFrameData)
    assert std.frame_detections_3d is not None
    ids = [d.unique_agent_id for d in std.frame_detections_3d.detections]
    assert ids == ["real-agent"], (
        "ghost (0,0,0) detection should have been skipped, only the real "
        f"detection emitted; got ids={ids}"
    )
    assert any(
        "ghost laser label" in record.message.lower() for record in caplog.records
    ), "expected a warning log for the skipped ghost label"


def test_detections_3d_adapter_wired_through_to_aggregator(tmp_path: Path):
    """End-to-end: synthesize 3 frames with one persistent agent, run the
    Detections3DIdentityAdapter and FutureDetectionsAggregator. Without the
    adapter wired, FutureDetectionsAggregator would read None for DETECTIONS_3D.
    """
    adapter = Detections3DIdentityAdapter()
    timestamps = [0.0, 0.1, 0.2]
    agent_xs = [10.0, 11.0, 12.0]
    frames: list[TransformedFrameData] = []
    for i, (ts, x) in enumerate(zip(timestamps, agent_xs)):
        std = _build_standard_frame("segA", i, ts, x)
        modalities = adapter.transform(std)
        frame = TransformedFrameData(
            dataset_name=std.dataset_name,
            segment_id=std.segment_id,
            frame_id=std.frame_id,
            timestamp=std.timestamp,
            split=std.split,
            global_position=std.global_position,
            aux_data=std.aux_data,
            _modality_data=modalities,
        )
        out_path = tmp_path / frame.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_npz(str(out_path))
        frames.append(frame)

    index_df = pd.DataFrame(
        {
            "segment_id": [f.segment_id for f in frames],
            "timestamp": [f.timestamp for f in frames],
            "filename": [f.filename for f in frames],
        }
    )

    aggr = FutureDetectionsAggregator(str(tmp_path))
    aggr.process(index_df)

    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, frames[0].filename))
    aggregated = reloaded.get_modality_data(Modality.DETECTIONS_3D)
    assert isinstance(aggregated, list), "aggregator output should be list[Detection3D]"
    assert len(aggregated) == 1, "exactly one persistent agent across the 3 frames"
    agent = aggregated[0]
    assert isinstance(agent, Detection3D)
    assert agent.unique_agent_id == "agent-0"
    # Two future frames -> trajectory length 2.
    assert agent.trajectory.length == 2
