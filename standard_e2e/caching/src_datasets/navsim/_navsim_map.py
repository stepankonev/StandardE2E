"""nuPlan vector map → unified ``MapElementType`` translation for NAVSIM.

NAVSIM ships HD maps as nuPlan ``map.gpkg`` files (one per city). The
``nuplan.common.maps.nuplan_map.map_factory.get_maps_api`` factory loads
a ``NuPlanMap`` that exposes 9 semantic layers; this module queries
within an ego-centric ROI and translates each map object into one or
more :class:`~standard_e2e.data_structures.MapElement` instances in the
ego frame.

Translation rules
-----------------

==============================  =====================================
nuPlan layer                    Unified ``MapElementType``
==============================  =====================================
``LANE``                        ``LANE_CENTER`` (baseline_path) +
                                ``LANE_BOUNDARY`` (left + right, no
                                paint info — nuPlan doesn't store
                                paint colour / pattern)
``LANE_CONNECTOR``              ``LANE_CENTER`` with
                                ``is_intersection=True``
``ROADBLOCK`` /
``ROADBLOCK_CONNECTOR``         ``DRIVABLE_AREA`` polygon
``INTERSECTION``                ``INTERSECTION`` polygon
``STOP_LINE``                   ``STOP_LINE`` polygon
``CROSSWALK``                   ``CROSSWALK`` polygon
``WALKWAYS``                    ``WALKWAY`` polygon
``CARPARK_AREA``                skipped
==============================  =====================================

``ROAD_EDGE`` / ``SPEED_BUMP`` / ``DRIVEWAY`` / ``TRAFFIC_LIGHT`` have no
nuPlan counterpart and are not emitted. ``TRAFFIC_LIGHT`` would come
from each frame's dynamic ``traffic_lights`` field, not the static map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from standard_e2e.data_structures import HDMap, MapElement
from standard_e2e.enums import MapElementType

if TYPE_CHECKING:  # pragma: no cover - import-only typing
    from nuplan.common.maps.abstract_map import AbstractMap


def _xy_city_to_ego(
    coords_city: np.ndarray,
    T_ego_from_city: np.ndarray,
    z_city: float = 0.0,
) -> np.ndarray:
    """Transform ``(N, 2)`` city XY to ``(N, 2)`` float32 ego XY.

    nuPlan maps are 2D, but the city frame is global UTM-like with city
    elevation in the hundreds of metres (e.g. Las Vegas ≈ 621 m). Even
    sub-degree ego pitch / roll then couples ``z_city`` into ``ego_x`` /
    ``ego_y`` through the off-diagonal terms ``R_inv[0, 2]`` and
    ``R_inv[1, 2]`` of the 4×4 inverse transform — a systematic
    ~10-meter lateral shift. Pass ``z_city = ego_z_global`` (the ego's
    altitude in city frame) so that ``(z_city − t_z) = 0`` and the
    off-diagonal coupling vanishes; the polygon then lies on the ego's
    local horizontal plane, which is what we want for a top-down BEV.
    """
    coords_city = np.asarray(coords_city, dtype=np.float64)
    n = coords_city.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    homo = np.column_stack(
        [
            coords_city,
            np.full(n, z_city, dtype=np.float64),
            np.ones(n, dtype=np.float64),
        ]
    )
    ego_homo = (np.asarray(T_ego_from_city, dtype=np.float64) @ homo.T).T
    return cast(np.ndarray, ego_homo[:, :2].astype(np.float32))


def _polygon_xy_to_ego(
    polygon: Any, T_ego_from_city: np.ndarray, z_city: float = 0.0
) -> np.ndarray:
    """Extract a shapely Polygon's exterior ring and transform to ego XY."""
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
    return _xy_city_to_ego(coords, T_ego_from_city, z_city=z_city)


def _linestring_xy_to_ego(
    linestring: Any, T_ego_from_city: np.ndarray, z_city: float = 0.0
) -> np.ndarray:
    """Extract a shapely LineString's coords and transform to ego XY."""
    coords = np.asarray(linestring.coords, dtype=np.float64)[:, :2]
    return _xy_city_to_ego(coords, T_ego_from_city, z_city=z_city)


def build_navsim_hd_map(
    nuplan_map: "AbstractMap",
    T_ego_from_city: np.ndarray,
    ego_x_city: float,
    ego_y_city: float,
    ego_z_city: float = 0.0,
    radius_m: float = 64.0,
) -> HDMap:
    """Query nuPlan within ``radius_m`` of the ego, translate to unified taxonomy.

    Args:
        nuplan_map: ``NuPlanMap`` instance (one per city).
        T_ego_from_city: 4×4 SE(3) transforming city coords into ego-frame coords.
        ego_x_city, ego_y_city: ego origin in city xy.
        ego_z_city: ego altitude in city frame. Pass the ego's actual z so
            map polygons are anchored to the ego's local horizontal plane;
            otherwise sub-degree pitch / roll couple the city elevation
            (~600 m for nuPlan cities) into a multi-meter lateral shift.
        radius_m: query radius around ego in meters.

    Returns:
        :class:`HDMap` with elements in the ego frame, ready for
        :class:`~standard_e2e.caching.adapters.HDMapBEVAdapter` rasterisation.
    """
    # Imports kept inside the function so this module imports cleanly even
    # when the ``nuplan-devkit`` extras aren't installed (e.g., docs build).
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    layers_to_query = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.WALKWAYS,
    ]
    proximal = nuplan_map.get_proximal_map_objects(
        Point2D(ego_x_city, ego_y_city), radius=radius_m, layers=layers_to_query
    )
    elements: list[MapElement] = []

    # ---- LANE → LANE_CENTER + 2 × LANE_BOUNDARY (paint absent) -----------
    for lane in proximal.get(SemanticMapLayer.LANE, []):
        center_attrs: dict[str, Any] = {
            "is_intersection": False,
            "lane_type": "vehicle",
        }
        if lane.speed_limit_mps is not None:
            center_attrs["speed_limit_mps"] = float(lane.speed_limit_mps)
        center_xy = _linestring_xy_to_ego(
            lane.baseline_path.linestring, T_ego_from_city, z_city=ego_z_city
        )
        elements.append(
            MapElement(
                id=f"lane_center_{lane.id}",
                type=MapElementType.LANE_CENTER,
                points=center_xy,
                is_closed=False,
                attrs=center_attrs,
            )
        )
        # nuPlan stores no paint info; emit boundaries with paint_*=None so
        # downstream consumers can still draw them (and so the schema matches
        # AV2 / Waymo, which DO emit paint).
        for side, boundary in (
            ("left", lane.left_boundary),
            ("right", lane.right_boundary),
        ):
            xy = _linestring_xy_to_ego(
                boundary.linestring, T_ego_from_city, z_city=ego_z_city
            )
            elements.append(
                MapElement(
                    id=f"lane_boundary_{lane.id}_{side}",
                    type=MapElementType.LANE_BOUNDARY,
                    points=xy,
                    is_closed=False,
                    attrs={"paint_color": None, "paint_pattern": None},
                )
            )

    # ---- LANE_CONNECTOR → LANE_CENTER w/ is_intersection=True -----------
    for conn in proximal.get(SemanticMapLayer.LANE_CONNECTOR, []):
        center_xy = _linestring_xy_to_ego(
            conn.baseline_path.linestring, T_ego_from_city, z_city=ego_z_city
        )
        elements.append(
            MapElement(
                id=f"lane_center_conn_{conn.id}",
                type=MapElementType.LANE_CENTER,
                points=center_xy,
                is_closed=False,
                attrs={"is_intersection": True, "lane_type": "vehicle"},
            )
        )

    # ---- ROADBLOCK + ROADBLOCK_CONNECTOR → DRIVABLE_AREA ----------------
    for layer in (
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ):
        for obj in proximal.get(layer, []):
            xy = _polygon_xy_to_ego(obj.polygon, T_ego_from_city, z_city=ego_z_city)
            elements.append(
                MapElement(
                    id=f"drivable_{layer.name}_{obj.id}",
                    type=MapElementType.DRIVABLE_AREA,
                    points=xy,
                    is_closed=True,
                )
            )

    # ---- INTERSECTION ---------------------------------------------------
    for inter in proximal.get(SemanticMapLayer.INTERSECTION, []):
        xy = _polygon_xy_to_ego(inter.polygon, T_ego_from_city, z_city=ego_z_city)
        elements.append(
            MapElement(
                id=f"intersection_{inter.id}",
                type=MapElementType.INTERSECTION,
                points=xy,
                is_closed=True,
            )
        )

    # ---- STOP_LINE ------------------------------------------------------
    for sl in proximal.get(SemanticMapLayer.STOP_LINE, []):
        xy = _polygon_xy_to_ego(sl.polygon, T_ego_from_city, z_city=ego_z_city)
        elements.append(
            MapElement(
                id=f"stop_line_{sl.id}",
                type=MapElementType.STOP_LINE,
                points=xy,
                is_closed=True,
            )
        )

    # ---- CROSSWALK ------------------------------------------------------
    for cw in proximal.get(SemanticMapLayer.CROSSWALK, []):
        xy = _polygon_xy_to_ego(cw.polygon, T_ego_from_city, z_city=ego_z_city)
        elements.append(
            MapElement(
                id=f"crosswalk_{cw.id}",
                type=MapElementType.CROSSWALK,
                points=xy,
                is_closed=True,
            )
        )

    # ---- WALKWAYS → WALKWAY --------------------------------------------
    for w in proximal.get(SemanticMapLayer.WALKWAYS, []):
        xy = _polygon_xy_to_ego(w.polygon, T_ego_from_city, z_city=ego_z_city)
        elements.append(
            MapElement(
                id=f"walkway_{w.id}",
                type=MapElementType.WALKWAY,
                points=xy,
                is_closed=True,
            )
        )

    return HDMap(elements=elements)
