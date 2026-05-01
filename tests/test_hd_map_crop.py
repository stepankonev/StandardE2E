"""Tests for ``crop_hd_map_ego_relative`` (PR 2 step 2.4b).

The crop helper is the **only** function that bridges the world-frame
``RawSegmentHDMap`` and the ego-frame ``HDMapData`` (per ADR 0006). All
its tests are pure-numpy synthetic, no source data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from standard_e2e.data_structures import (
    Crosswalk,
    DrivableArea,
    HDMapData,
    Lane,
    LaneBoundary,
    RawSegmentHDMap,
    RoadEdge,
    StopSign,
)
from standard_e2e.enums import LaneMarkType, LaneType, RoadEdgeType
from standard_e2e.utils.hd_map import crop_hd_map_ego_relative


def _ego_pose_at(x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    """world<-ego SE(3): ego origin at (x, y, 0) with given yaw."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = cos_y
    pose[0, 1] = -sin_y
    pose[1, 0] = sin_y
    pose[1, 1] = cos_y
    pose[0, 3] = x
    pose[1, 3] = y
    return pose


def test_crop_drops_polylines_entirely_outside_box():
    """Only one of four polylines stays after a 50x50 m crop."""
    inside = Lane(
        centerline=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        lane_type=LaneType.VEHICLE,
    )
    far_x = Lane(
        centerline=np.array([[1000, 0, 0], [1010, 0, 0]], dtype=np.float32),
        lane_type=LaneType.VEHICLE,
    )
    far_y = Lane(
        centerline=np.array([[0, 1000, 0], [0, 1010, 0]], dtype=np.float32),
        lane_type=LaneType.VEHICLE,
    )
    far_neg = Lane(
        centerline=np.array([[-200, -200, 0], [-210, -210, 0]], dtype=np.float32),
        lane_type=LaneType.VEHICLE,
    )
    raw = RawSegmentHDMap(lanes=[inside, far_x, far_y, far_neg])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert isinstance(out, HDMapData)
    assert len(out.lanes) == 1


def test_crop_translates_and_rotates_polylines_into_ego_frame():
    """A lane whose world coords are (10, 5) lifts to ego coords (0, -5)
    when the ego is at world (10, 10) with yaw=0."""
    lane = Lane(
        centerline=np.array([[10, 5, 0], [12, 5, 0]], dtype=np.float32),
        lane_type=LaneType.VEHICLE,
    )
    raw = RawSegmentHDMap(lanes=[lane])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(10.0, 10.0), 50.0, 50.0)
    assert len(out.lanes) == 1
    expected = np.array([[0, -5, 0], [2, -5, 0]], dtype=np.float32)
    np.testing.assert_allclose(out.lanes[0].centerline, expected, atol=1e-5)


def test_crop_keeps_polygon_with_any_vertex_inside():
    inside_vertex = Crosswalk(
        polygon=np.array([[0, 0, 0], [200, 0, 0], [200, 200, 0], [0, 200, 0]], dtype=np.float32)
    )
    fully_outside = Crosswalk(
        polygon=np.array(
            [[200, 200, 0], [210, 200, 0], [210, 210, 0], [200, 210, 0]], dtype=np.float32
        )
    )
    raw = RawSegmentHDMap(crosswalks=[inside_vertex, fully_outside])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert len(out.crosswalks) == 1


def test_crop_filters_polyline_vertices_to_those_inside_box():
    """LaneBoundary points outside the box are filtered; min 2 remaining."""
    boundary = LaneBoundary(
        polyline=np.array(
            [[-1000, 0, 0], [0, 0, 0], [10, 0, 0], [1000, 0, 0]], dtype=np.float32
        ),
        boundary_type=LaneMarkType.SOLID_WHITE,
    )
    raw = RawSegmentHDMap(lane_boundaries=[boundary])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert len(out.lane_boundaries) == 1
    assert out.lane_boundaries[0].polyline.shape == (2, 3)


def test_crop_drops_polyline_with_fewer_than_two_inside_vertices():
    boundary = LaneBoundary(
        polyline=np.array([[0, 0, 0], [1000, 0, 0]], dtype=np.float32),
        boundary_type=LaneMarkType.SOLID_WHITE,
    )
    raw = RawSegmentHDMap(lane_boundaries=[boundary])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert out.lane_boundaries == []


def test_crop_includes_drivable_area_with_any_vertex_inside():
    polygon = np.array(
        [[1000, 0, 0], [1100, 0, 0], [1100, 100, 0], [10, 10, 0]],
        dtype=np.float32,
    )
    raw = RawSegmentHDMap(drivable_areas=[DrivableArea(polygon=polygon)])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert len(out.drivable_areas) == 1


def test_crop_filters_stop_signs_outside_box():
    raw = RawSegmentHDMap(
        stop_signs=[
            StopSign(position=np.array([5, 0, 0], dtype=np.float32)),
            StopSign(position=np.array([1000, 0, 0], dtype=np.float32)),
        ]
    )
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert len(out.stop_signs) == 1
    np.testing.assert_allclose(
        out.stop_signs[0].position, np.array([5, 0, 0], dtype=np.float32)
    )


def test_crop_returns_hd_map_data_not_raw():
    """Type-distinctness: output is the ego type, not the world type."""
    raw = RawSegmentHDMap(road_edges=[
        RoadEdge(
            polyline=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            road_edge_type=RoadEdgeType.BOUNDARY,
        )
    ])
    out = crop_hd_map_ego_relative(raw, _ego_pose_at(0, 0), 50.0, 50.0)
    assert isinstance(out, HDMapData)
    assert not isinstance(out, RawSegmentHDMap)


def test_crop_rejects_hd_map_data_input():
    """Catches the 'tried to crop ego-frame data again' bug."""
    ego = HDMapData()
    with pytest.raises(TypeError):
        crop_hd_map_ego_relative(ego, _ego_pose_at(0, 0), 50.0, 50.0)


def test_raw_segment_hd_map_has_no_persistence_methods():
    """Negative regression guard: RawSegmentHDMap is runtime-only Pydantic.

    If anyone adds to_npz/from_npz to it, this test must fail so the
    reviewer notices the ADR 0007 invariant has changed.
    """
    assert not hasattr(RawSegmentHDMap, "to_npz")
    assert not hasattr(RawSegmentHDMap, "from_npz")
