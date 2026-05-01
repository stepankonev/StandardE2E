"""Parse a Waymo Perception ``Frame.map_features`` into a world-frame
``RawSegmentHDMap`` (per ADR 0006).

Waymo stores HD-map points in a per-segment "segment frame" that is
shifted from the world frame for numerical precision; the shift is the
``Frame.map_pose_offset`` Vector3d. Lifting to world is **translation
only**:

    world_point = map_point + (offset.x, offset.y, offset.z)

``Frame.map_pose_offset`` is a ``Vector3d``, **not** a 4x4 transform —
do not matrix-multiply it with ``Frame.pose``. ``Frame.pose`` is the
per-frame world<-ego pose used downstream by the ego-crop aggregator,
not by this parser.

The output is a ``RawSegmentHDMap`` (world frame). The per-source
aggregator passes it through ``crop_hd_map_ego_relative`` once per
frame to get the ego-frame ``HDMapData`` that is persisted to disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from standard_e2e.data_structures import (
    Crosswalk,
    Driveway,
    Lane,
    LaneBoundary,
    RawSegmentHDMap,
    RoadEdge,
    SpeedBump,
    StopSign,
)
from standard_e2e.enums import LaneMarkType, LaneType, RoadEdgeType

if TYPE_CHECKING:
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import (
        Frame as WaymoFrame,
    )


def _polyline_to_array(repeated_points, offset: np.ndarray) -> np.ndarray:
    if len(repeated_points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in repeated_points], dtype=np.float32)
    out: np.ndarray = pts + offset
    return out


def _waymo_lane_type(value: int) -> LaneType:
    # Imported lazily so this module is importable without TF in the
    # base install (LaneCenter belongs to the waymo extra).
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import LaneCenter

    return {
        LaneCenter.TYPE_UNDEFINED: LaneType.UNKNOWN,
        LaneCenter.TYPE_FREEWAY: LaneType.VEHICLE,
        LaneCenter.TYPE_SURFACE_STREET: LaneType.VEHICLE,
        LaneCenter.TYPE_BIKE_LANE: LaneType.BIKE,
    }.get(value, LaneType.UNKNOWN)


def _waymo_road_line_type(value: int) -> LaneMarkType:
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import RoadLine

    return {
        RoadLine.TYPE_UNKNOWN: LaneMarkType.UNKNOWN,
        RoadLine.TYPE_BROKEN_SINGLE_WHITE: LaneMarkType.DASHED_WHITE,
        RoadLine.TYPE_SOLID_SINGLE_WHITE: LaneMarkType.SOLID_WHITE,
        RoadLine.TYPE_SOLID_DOUBLE_WHITE: LaneMarkType.DOUBLE_SOLID_WHITE,
        RoadLine.TYPE_BROKEN_SINGLE_YELLOW: LaneMarkType.DASHED_YELLOW,
        RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: LaneMarkType.DASHED_YELLOW,
        RoadLine.TYPE_SOLID_SINGLE_YELLOW: LaneMarkType.SOLID_YELLOW,
        RoadLine.TYPE_SOLID_DOUBLE_YELLOW: LaneMarkType.DOUBLE_SOLID_YELLOW,
        RoadLine.TYPE_PASSING_DOUBLE_YELLOW: LaneMarkType.PASSING_DOUBLE_DASH,
    }.get(value, LaneMarkType.UNKNOWN)


def _waymo_road_edge_type(value: int) -> RoadEdgeType:
    # pylint: disable=no-name-in-module
    from standard_e2e.third_party.waymo_open_dataset.protos.map_pb2 import (
        RoadEdge as WaymoRoadEdge,
    )

    return {
        WaymoRoadEdge.TYPE_UNKNOWN: RoadEdgeType.UNKNOWN,
        WaymoRoadEdge.TYPE_ROAD_EDGE_BOUNDARY: RoadEdgeType.BOUNDARY,
        WaymoRoadEdge.TYPE_ROAD_EDGE_MEDIAN: RoadEdgeType.MEDIAN,
    }.get(value, RoadEdgeType.UNKNOWN)


def parse_waymo_map_features(frame: "WaymoFrame") -> RawSegmentHDMap:
    """Build a world-frame ``RawSegmentHDMap`` from one Waymo frame.

    For Waymo Perception v1.4.x, ``map_features`` are populated only on
    the first frame of each segment; this function does not assume
    that — it just reads ``frame.map_features`` and lifts every entry
    to world frame using ``frame.map_pose_offset``.
    """
    offset = np.array(
        [frame.map_pose_offset.x, frame.map_pose_offset.y, frame.map_pose_offset.z],
        dtype=np.float32,
    )

    lanes: list[Lane] = []
    lane_boundaries: list[LaneBoundary] = []
    road_edges: list[RoadEdge] = []
    crosswalks: list[Crosswalk] = []
    stop_signs: list[StopSign] = []
    speed_bumps: list[SpeedBump] = []
    driveways: list[Driveway] = []

    for feat in frame.map_features:
        if feat.HasField("lane"):
            centerline = _polyline_to_array(feat.lane.polyline, offset)
            if centerline.shape[0] < 1:
                continue
            # Boundary feature ids (opaque source IDs; per ADR 0006 we
            # keep them as identifiers for re-deduplication, not as a
            # full per-lane slice — that lossier dereferencing can land
            # in a follow-up).
            left_ids = [str(b.boundary_feature_id) for b in feat.lane.left_boundaries]
            right_ids = [str(b.boundary_feature_id) for b in feat.lane.right_boundaries]
            lanes.append(
                Lane(
                    centerline=centerline,
                    left_boundary_id=",".join(left_ids) if left_ids else None,
                    right_boundary_id=",".join(right_ids) if right_ids else None,
                    predecessors=[str(i) for i in feat.lane.entry_lanes],
                    successors=[str(i) for i in feat.lane.exit_lanes],
                    is_intersection=False,
                    lane_type=_waymo_lane_type(feat.lane.type),
                )
            )
        elif feat.HasField("road_line"):
            polyline = _polyline_to_array(feat.road_line.polyline, offset)
            if polyline.shape[0] < 2:
                continue
            lane_boundaries.append(
                LaneBoundary(
                    polyline=polyline,
                    boundary_type=_waymo_road_line_type(feat.road_line.type),
                    source_boundary_id=str(feat.id),
                )
            )
        elif feat.HasField("road_edge"):
            polyline = _polyline_to_array(feat.road_edge.polyline, offset)
            if polyline.shape[0] < 2:
                continue
            road_edges.append(
                RoadEdge(
                    polyline=polyline,
                    road_edge_type=_waymo_road_edge_type(feat.road_edge.type),
                )
            )
        elif feat.HasField("crosswalk"):
            polygon = _polyline_to_array(feat.crosswalk.polygon, offset)
            if polygon.shape[0] >= 3:
                crosswalks.append(Crosswalk(polygon=polygon))
        elif feat.HasField("stop_sign"):
            position = (
                np.array(
                    [
                        feat.stop_sign.position.x,
                        feat.stop_sign.position.y,
                        feat.stop_sign.position.z,
                    ],
                    dtype=np.float32,
                )
                + offset
            )
            stop_signs.append(
                StopSign(
                    position=position,
                    lane_ids=[str(i) for i in feat.stop_sign.lane],
                )
            )
        elif feat.HasField("speed_bump"):
            polygon = _polyline_to_array(feat.speed_bump.polygon, offset)
            if polygon.shape[0] >= 3:
                speed_bumps.append(SpeedBump(polygon=polygon))
        elif feat.HasField("driveway"):
            polygon = _polyline_to_array(feat.driveway.polygon, offset)
            if polygon.shape[0] >= 3:
                driveways.append(Driveway(polygon=polygon))

    return RawSegmentHDMap(
        lanes=lanes,
        lane_boundaries=lane_boundaries,
        road_edges=road_edges,
        crosswalks=crosswalks,
        stop_signs=stop_signs,
        speed_bumps=speed_bumps,
        drivable_areas=[],
        driveways=driveways,
    )
