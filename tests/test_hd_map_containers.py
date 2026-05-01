"""Tests for the HD-map data structures + identity adapter (PR 2 step 2.3).

These cover:
- ``HDMapData`` (ego frame) and ``RawSegmentHDMap`` (world frame) Pydantic
  model construction with all 8 element list fields.
- Type-distinctness: ``RawSegmentHDMap`` is **not** an ``HDMapData`` and
  vice versa (so consumers that expect the ego type cannot accidentally
  receive the world-frame intermediate).
- ``HDMapIdentityAdapter`` lifts ``standard_frame.hd_map`` into the
  ``Modality.HD_MAP`` slot.
- Collate over a 2-frame batch (passthrough list, lossless — same shape
  as the lidar collate).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from standard_e2e.caching.adapters import HDMapIdentityAdapter
from standard_e2e.data_structures import (
    StandardFrameData,
    TransformedFrameData,
)
from standard_e2e.data_structures.containers import (
    Crosswalk,
    DrivableArea,
    Driveway,
    HDMapData,
    Lane,
    LaneBoundary,
    RawSegmentHDMap,
    RoadEdge,
    SpeedBump,
    StopSign,
)
from standard_e2e.data_structures.frame_data import collate_modalities
from standard_e2e.enums import LaneMarkType, LaneType, Modality, RoadEdgeType


def _polyline_xyz(*pts: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(pts, dtype=np.float32)


def _build_full_field_kwargs() -> dict:
    """One element in every list. Same shape used for both HDMapData and
    RawSegmentHDMap."""
    lane = Lane(
        centerline=_polyline_xyz((0, 0, 0), (10, 0, 0)),
        left_boundary_id="lb-1",
        right_boundary_id="lb-2",
        predecessors=[],
        successors=["lane-2"],
        is_intersection=False,
        lane_type=LaneType.VEHICLE,
    )
    lane_boundary = LaneBoundary(
        polyline=_polyline_xyz((0, 1, 0), (10, 1, 0)),
        boundary_type=LaneMarkType.SOLID_WHITE,
        source_boundary_id="lb-1",
    )
    road_edge = RoadEdge(
        polyline=_polyline_xyz((0, 5, 0), (10, 5, 0)),
        road_edge_type=RoadEdgeType.BOUNDARY,
    )
    crosswalk = Crosswalk(polygon=_polyline_xyz((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    stop_sign = StopSign(position=np.array([5.0, 0.0, 0.0], dtype=np.float32))
    speed_bump = SpeedBump(polygon=_polyline_xyz((2, -1, 0), (2, 1, 0), (3, 1, 0), (3, -1, 0)))
    drivable_area = DrivableArea(polygon=_polyline_xyz((0, -2, 0), (10, -2, 0), (10, 2, 0), (0, 2, 0)))
    driveway = Driveway(polygon=_polyline_xyz((-2, 0, 0), (-2, 2, 0), (0, 2, 0), (0, 0, 0)))
    return dict(
        lanes=[lane],
        lane_boundaries=[lane_boundary],
        road_edges=[road_edge],
        crosswalks=[crosswalk],
        stop_signs=[stop_sign],
        speed_bumps=[speed_bump],
        drivable_areas=[drivable_area],
        driveways=[driveway],
    )


def test_hd_map_data_full_field_construction():
    m = HDMapData(**_build_full_field_kwargs())
    assert len(m.lanes) == 1
    assert len(m.lane_boundaries) == 1
    assert len(m.road_edges) == 1
    assert len(m.crosswalks) == 1
    assert len(m.stop_signs) == 1
    assert len(m.speed_bumps) == 1
    assert len(m.drivable_areas) == 1
    assert len(m.driveways) == 1


def test_hd_map_data_defaults_to_empty_lists():
    m = HDMapData()
    assert m.lanes == []
    assert m.lane_boundaries == []
    assert m.road_edges == []
    assert m.crosswalks == []
    assert m.stop_signs == []
    assert m.speed_bumps == []
    assert m.drivable_areas == []
    assert m.driveways == []


def test_raw_segment_hd_map_full_field_construction():
    raw = RawSegmentHDMap(**_build_full_field_kwargs())
    assert len(raw.lanes) == 1
    assert len(raw.drivable_areas) == 1


def test_hd_map_data_and_raw_are_distinct_types():
    """An HDMapData is NOT a RawSegmentHDMap, even though they share fields."""
    ego = HDMapData(**_build_full_field_kwargs())
    raw = RawSegmentHDMap(**_build_full_field_kwargs())
    assert not isinstance(ego, RawSegmentHDMap)
    assert not isinstance(raw, HDMapData)


def test_hd_map_data_persists_through_npz_roundtrip(tmp_path: Path):
    """Ego HDMapData survives the standard TransformedFrameData .npz path."""
    ego = HDMapData(**_build_full_field_kwargs())
    frame = TransformedFrameData(
        dataset_name="ds",
        split="train",
        segment_id="segH",
        frame_id=0,
        timestamp=0.0,
    )
    frame.set_modality_data(Modality.HD_MAP, ego)
    out = tmp_path / frame.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_npz(str(out))

    reloaded = TransformedFrameData.from_npz(os.path.join(tmp_path, frame.filename))
    payload = reloaded.get_modality_data(Modality.HD_MAP)
    assert isinstance(payload, HDMapData)
    assert len(payload.lanes) == 1
    np.testing.assert_array_equal(
        payload.lanes[0].centerline, ego.lanes[0].centerline
    )


def test_hd_map_identity_adapter_lifts_field_into_modality():
    ego = HDMapData(**_build_full_field_kwargs())
    std = StandardFrameData(
        dataset_name="ds",
        split="train",
        segment_id="segH",
        frame_id=0,
        timestamp=0.0,
        hd_map=ego,
    )
    adapter = HDMapIdentityAdapter()
    out = adapter.transform(std)
    assert Modality.HD_MAP in out
    assert out[Modality.HD_MAP] is ego


def test_hd_map_identity_adapter_no_hd_map():
    std = StandardFrameData(
        dataset_name="ds",
        split="train",
        segment_id="segH",
        frame_id=0,
        timestamp=0.0,
    )
    adapter = HDMapIdentityAdapter()
    assert adapter.transform(std) == {}


def test_collate_hd_map_returns_passthrough_list():
    a = HDMapData(**_build_full_field_kwargs())
    b = HDMapData()
    collated = collate_modalities([a, b])
    assert isinstance(collated, list)
    assert len(collated) == 2
    assert collated[0] is a
    assert collated[1] is b
