"""HD-map crop helper: world-frame -> ego-frame conversion (per ADR 0006).

This module exposes exactly one public function,
``crop_hd_map_ego_relative``: the **single bridge** between the
world-frame ``RawSegmentHDMap`` and the ego-frame ``HDMapData``.
Splitting the two coord-frame variants into distinct types means this
function is the only place the world -> ego transform happens.

Geometry conventions:
- ``ego_pose`` is the 4x4 SE(3) ``world<-ego`` transform (per
  ``CONTEXT.md``). Conversion is ``inv(ego_pose) @ p_world_h``.
- ``x_range`` / ``y_range`` are **half-extents** in meters (so
  ``x_range=50`` keeps points with ``-50 <= x <= 50``).
- Lanes: drop entire lane if no centerline vertex is inside the box;
  otherwise keep the lane with the full centerline transformed (so the
  consumer still has continuous geometry across the box edge).
- Polylines (``LaneBoundary``, ``RoadEdge``): vertex-filter to those
  inside the box. Drop if fewer than 2 vertices remain. This is a
  deliberate simplification of full Cohen-Sutherland-style line
  clipping; at typical map polyline density a vertex filter is within a
  few meters of true clipping and avoids introducing fake vertices on
  the box boundary.
- Polygons (``Crosswalk``, ``SpeedBump``, ``DrivableArea``,
  ``Driveway``): include the whole polygon (in ego frame) if **any**
  vertex falls inside the box. This is conservative: it preserves
  partially-overlapping polygons rather than clipping them to the box,
  which avoids re-triangulating concave shapes.
- Stop signs: kept iff the (single) point is inside the box.
"""

from __future__ import annotations

import numpy as np

from standard_e2e.data_structures import (
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


def _world_to_ego(points_world: np.ndarray, inv_ego_pose: np.ndarray) -> np.ndarray:
    """Transform (N, 3) world-frame points into the ego frame.

    Takes the precomputed ``inv_ego_pose`` (``ego<-world``) so callers
    that lift many element batches under one ego pose pay one inversion
    instead of one per call.
    """
    if points_world.size == 0:
        return points_world.astype(np.float32, copy=False)
    homog = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=points_world.dtype)],
        axis=1,
    )
    ego = (inv_ego_pose @ homog.T).T
    out: np.ndarray = ego[:, :3].astype(np.float32, copy=False)
    return out


def _within_box(points: np.ndarray, x_range: float, y_range: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points[:, 0] >= -x_range)
        & (points[:, 0] <= x_range)
        & (points[:, 1] >= -y_range)
        & (points[:, 1] <= y_range)
    )


def crop_hd_map_ego_relative(
    raw: RawSegmentHDMap,
    ego_pose: np.ndarray,
    x_range: float,
    y_range: float,
) -> HDMapData:
    """Convert a segment-wide world-frame map into an ego-frame crop.

    Args:
        raw: World-frame ``RawSegmentHDMap``. Passing an ``HDMapData``
            (already ego-frame) raises ``TypeError`` — see ADR 0006 on
            why the two are distinct types.
        ego_pose: 4x4 ``world<-ego`` SE(3) at the current frame's
            timestamp.
        x_range: half-extent in meters along the ego x axis.
        y_range: half-extent in meters along the ego y axis.

    Returns:
        ``HDMapData`` with only the elements that intersect the box, in
        ego frame.
    """
    if not isinstance(raw, RawSegmentHDMap):
        raise TypeError(
            "crop_hd_map_ego_relative expects RawSegmentHDMap (world frame); "
            f"got {type(raw).__name__}. HDMapData is already in ego frame; "
            "do not re-crop."
        )
    if ego_pose.shape != (4, 4):
        raise ValueError(f"ego_pose must be (4,4); got {ego_pose.shape}")

    inv_ego = np.linalg.inv(ego_pose).astype(np.float32, copy=False)

    lanes_out: list[Lane] = []
    for lane in raw.lanes:
        ego_centerline = _world_to_ego(lane.centerline, inv_ego)
        if _within_box(ego_centerline, x_range, y_range).any():
            lanes_out.append(
                Lane(
                    centerline=ego_centerline,
                    left_boundary_id=lane.left_boundary_id,
                    right_boundary_id=lane.right_boundary_id,
                    predecessors=list(lane.predecessors),
                    successors=list(lane.successors),
                    is_intersection=lane.is_intersection,
                    lane_type=lane.lane_type,
                )
            )

    boundaries_out: list[LaneBoundary] = []
    for b in raw.lane_boundaries:
        ego_pts = _world_to_ego(b.polyline, inv_ego)
        mask = _within_box(ego_pts, x_range, y_range)
        if int(mask.sum()) >= 2:
            boundaries_out.append(
                LaneBoundary(
                    polyline=ego_pts[mask],
                    boundary_type=b.boundary_type,
                    source_boundary_id=b.source_boundary_id,
                )
            )

    edges_out: list[RoadEdge] = []
    for e in raw.road_edges:
        ego_pts = _world_to_ego(e.polyline, inv_ego)
        mask = _within_box(ego_pts, x_range, y_range)
        if int(mask.sum()) >= 2:
            edges_out.append(
                RoadEdge(polyline=ego_pts[mask], road_edge_type=e.road_edge_type)
            )

    def _crop_polygon_list(items: list, cls):
        out: list = []
        for item in items:
            ego_pts = _world_to_ego(item.polygon, inv_ego)
            if _within_box(ego_pts, x_range, y_range).any():
                out.append(cls(polygon=ego_pts))
        return out

    crosswalks_out = _crop_polygon_list(raw.crosswalks, Crosswalk)
    speed_bumps_out = _crop_polygon_list(raw.speed_bumps, SpeedBump)
    drivable_areas_out = _crop_polygon_list(raw.drivable_areas, DrivableArea)
    driveways_out = _crop_polygon_list(raw.driveways, Driveway)

    stop_signs_out: list[StopSign] = []
    for s in raw.stop_signs:
        ego_pos = _world_to_ego(s.position[None, :], inv_ego)[0]
        in_x = -x_range <= float(ego_pos[0]) <= x_range
        in_y = -y_range <= float(ego_pos[1]) <= y_range
        if in_x and in_y:
            stop_signs_out.append(StopSign(position=ego_pos, lane_ids=list(s.lane_ids)))

    return HDMapData(
        lanes=lanes_out,
        lane_boundaries=boundaries_out,
        road_edges=edges_out,
        crosswalks=crosswalks_out,
        stop_signs=stop_signs_out,
        speed_bumps=speed_bumps_out,
        drivable_areas=drivable_areas_out,
        driveways=driveways_out,
    )
