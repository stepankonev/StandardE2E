"""Tests for ``parse_waymo_map_features`` (PR 2 step 2.5).

The parser converts a Waymo ``Frame.map_features`` collection (in
segment frame) into a world-frame ``RawSegmentHDMap``. Lift to world is
**translation only** by ``Frame.map_pose_offset`` (a ``Vector3d``, not
a 4x4 transform — see ADR 0006 / step 2.5 spec).

Tests synthesize a Waymo ``Frame`` proto in memory; no real tfrecord on
disk required.
"""

from __future__ import annotations

import numpy as np

# pylint: disable=no-name-in-module
from standard_e2e.caching.src_datasets.waymo_perception.hd_map_parser import (
    parse_waymo_map_features,
)
from standard_e2e.data_structures import RawSegmentHDMap
from standard_e2e.enums import LaneMarkType, LaneType, RoadEdgeType
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame
from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import (
    LaneCenter,
    RoadLine,
)


def _add_polyline(repeated_points, pts: list[tuple[float, float, float]]):
    for x, y, z in pts:
        p = repeated_points.add()
        p.x = x
        p.y = y
        p.z = z


def _frame_with_offset(ox: float = 0.0, oy: float = 0.0, oz: float = 0.0) -> Frame:
    f = Frame()
    f.map_pose_offset.x = ox
    f.map_pose_offset.y = oy
    f.map_pose_offset.z = oz
    return f


def test_parser_returns_raw_segment_hd_map_type():
    f = _frame_with_offset()
    out = parse_waymo_map_features(f)
    assert isinstance(out, RawSegmentHDMap)


def test_parser_lane_centerline_lifted_to_world_by_offset():
    """Segment-frame point + map_pose_offset == world-frame point."""
    f = _frame_with_offset(100.0, 200.0, 5.0)
    feat = f.map_features.add()
    feat.id = 1
    feat.lane.type = LaneCenter.TYPE_SURFACE_STREET
    _add_polyline(feat.lane.polyline, [(0, 0, 0), (10, 0, 0)])

    out = parse_waymo_map_features(f)
    assert len(out.lanes) == 1
    expected = np.array([[100, 200, 5], [110, 200, 5]], dtype=np.float32)
    np.testing.assert_allclose(out.lanes[0].centerline, expected, atol=1e-5)
    assert out.lanes[0].lane_type == LaneType.VEHICLE


def test_parser_lane_type_mapping():
    f = _frame_with_offset()
    for waymo_type, expected in (
        (LaneCenter.TYPE_UNDEFINED, LaneType.UNKNOWN),
        (LaneCenter.TYPE_FREEWAY, LaneType.VEHICLE),
        (LaneCenter.TYPE_SURFACE_STREET, LaneType.VEHICLE),
        (LaneCenter.TYPE_BIKE_LANE, LaneType.BIKE),
    ):
        f.ClearField("map_features")
        feat = f.map_features.add()
        feat.id = 1
        feat.lane.type = waymo_type
        _add_polyline(feat.lane.polyline, [(0, 0, 0), (1, 0, 0)])
        out = parse_waymo_map_features(f)
        assert out.lanes[0].lane_type == expected


def test_parser_road_line_becomes_lane_boundary():
    f = _frame_with_offset(10.0, 0.0, 0.0)
    feat = f.map_features.add()
    feat.id = 42
    feat.road_line.type = RoadLine.TYPE_SOLID_SINGLE_WHITE
    _add_polyline(feat.road_line.polyline, [(0, 1, 0), (5, 1, 0)])

    out = parse_waymo_map_features(f)
    assert len(out.lane_boundaries) == 1
    boundary = out.lane_boundaries[0]
    assert boundary.boundary_type == LaneMarkType.SOLID_WHITE
    assert boundary.source_boundary_id == "42"
    np.testing.assert_allclose(
        boundary.polyline, np.array([[10, 1, 0], [15, 1, 0]], dtype=np.float32), atol=1e-5
    )


def test_parser_road_edge_field_populated():
    from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import RoadEdge

    f = _frame_with_offset()
    feat = f.map_features.add()
    feat.id = 99
    feat.road_edge.type = RoadEdge.TYPE_ROAD_EDGE_BOUNDARY
    _add_polyline(feat.road_edge.polyline, [(0, 5, 0), (10, 5, 0)])

    out = parse_waymo_map_features(f)
    assert len(out.road_edges) == 1
    assert out.road_edges[0].road_edge_type == RoadEdgeType.BOUNDARY


def test_parser_crosswalk_polygon():
    f = _frame_with_offset(50.0, 0.0, 0.0)
    feat = f.map_features.add()
    feat.id = 7
    _add_polyline(
        feat.crosswalk.polygon,
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    )
    out = parse_waymo_map_features(f)
    assert len(out.crosswalks) == 1
    np.testing.assert_allclose(
        out.crosswalks[0].polygon[0], np.array([50, 0, 0], dtype=np.float32), atol=1e-5
    )


def test_parser_stop_sign_position():
    f = _frame_with_offset(0.0, 0.0, 0.0)
    feat = f.map_features.add()
    feat.id = 5
    feat.stop_sign.position.x = 10.0
    feat.stop_sign.position.y = 5.0
    feat.stop_sign.position.z = 0.0
    feat.stop_sign.lane.append(11)
    feat.stop_sign.lane.append(12)
    out = parse_waymo_map_features(f)
    assert len(out.stop_signs) == 1
    np.testing.assert_allclose(
        out.stop_signs[0].position, np.array([10, 5, 0], dtype=np.float32), atol=1e-5
    )
    assert out.stop_signs[0].lane_ids == ["11", "12"]


def test_parser_speed_bump_and_driveway():
    f = _frame_with_offset()
    feat_sb = f.map_features.add()
    feat_sb.id = 100
    _add_polyline(feat_sb.speed_bump.polygon, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])

    feat_dw = f.map_features.add()
    feat_dw.id = 101
    _add_polyline(feat_dw.driveway.polygon, [(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)])

    out = parse_waymo_map_features(f)
    assert len(out.speed_bumps) == 1
    assert len(out.driveways) == 1


def test_parser_lane_neighbors_populated_from_entry_exit_lanes():
    f = _frame_with_offset()
    feat = f.map_features.add()
    feat.id = 10
    feat.lane.type = LaneCenter.TYPE_SURFACE_STREET
    _add_polyline(feat.lane.polyline, [(0, 0, 0), (1, 0, 0)])
    feat.lane.entry_lanes.append(7)
    feat.lane.entry_lanes.append(8)
    feat.lane.exit_lanes.append(20)

    out = parse_waymo_map_features(f)
    assert out.lanes[0].predecessors == ["7", "8"]
    assert out.lanes[0].successors == ["20"]


def test_parser_empty_frame_yields_empty_map():
    f = _frame_with_offset()
    out = parse_waymo_map_features(f)
    assert out.lanes == []
    assert out.lane_boundaries == []
    assert out.crosswalks == []
