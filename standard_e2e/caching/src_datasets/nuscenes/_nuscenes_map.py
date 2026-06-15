"""nuScenes vector map (map-expansion) -> unified ``HDMap``, read directly.

The nuScenes **map-expansion** pack ships one ``maps/expansion/<location>.json``
per map location: a normalized vector map of geometry primitives (``node`` /
``line`` / ``polygon``) plus semantic layers that reference them. We can't use
the devkit's ``NuScenesMap`` API (it pins ``numpy<2`` against this project's
numpy 2.x; see the lane-centerline note in :mod:`._nuscenes_arcline`), so this
module parses the JSON directly and translates each layer into one or more
:class:`~standard_e2e.data_structures.MapElement` instances, following the
cross-dataset taxonomy in ``docs/map_taxonomy.md``:

================  ===================================================
nuScenes layer    Unified ``MapElementType``
================  ===================================================
``lane``          ``LANE_CENTER`` (centerline from the arcline path)
``lane_connector``  ``LANE_CENTER`` with ``attrs.is_intersection=True``
``lane_divider``  ``LANE_BOUNDARY`` (paint sub-type not shipped -> None)
``road_divider``  ``ROAD_EDGE`` (``road_edge_subtype="median"``)
``ped_crossing``  ``CROSSWALK`` (polygon)
``walkway``       ``WALKWAY`` (polygon)
``stop_line``     ``STOP_LINE`` (polygon, ``attrs.stop_line_type``)
``drivable_area`` ``DRIVABLE_AREA`` (one element per polygon)
``road_segment``  ``INTERSECTION`` (polygon, only where ``is_intersection``)
================  ===================================================

``carpark_area`` and the static ``traffic_light`` layer have no target yet
(traffic-light *state* is dynamic and not in the static map), matching how the
NAVSIM/nuPlan processor skips carparks and dynamic lights.

A :class:`NuscMap` is parsed once per location (cached by the processor) into
global-frame elements with a bounding circle each; :meth:`NuscMap.build_hd_map`
then ROI-filters and transforms them into a single frame's ego coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from standard_e2e.caching.src_datasets.nuscenes._nuscenes_arcline import discretize_lane
from standard_e2e.data_structures import HDMap, MapElement
from standard_e2e.enums import MapElementType

# Lane centerline discretization step (m) and default ego-centric query radius.
_LANE_RESOLUTION_M = 1.0
_DEFAULT_RADIUS_M = 64.0

# nuScenes lane_type -> unified attrs.lane_type (docs/map_taxonomy.md §8).
_LANE_TYPE = {"CAR": "vehicle", "BIKE": "bike", "BUS": "bus"}


@dataclass(frozen=True)
class _GlobalElement:
    """One map element in global (map) coordinates, with a bounding circle for
    cheap ROI prefiltering."""

    id: str
    type: MapElementType
    points: np.ndarray  # (N, 2) float64, global
    is_closed: bool
    attrs: dict[str, Any]
    successor_ids: tuple[str, ...]
    predecessor_ids: tuple[str, ...]
    center: np.ndarray  # (2,) float64
    bound_radius: float


def _make_element(
    element_id: str,
    element_type: MapElementType,
    points: np.ndarray,
    is_closed: bool,
    attrs: Optional[dict[str, Any]] = None,
    successor_ids: tuple[str, ...] = (),
    predecessor_ids: tuple[str, ...] = (),
) -> Optional[_GlobalElement]:
    """Build a ``_GlobalElement`` (None if the geometry is empty)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return None
    center = pts.mean(axis=0)
    bound_radius = float(np.linalg.norm(pts - center, axis=1).max())
    return _GlobalElement(
        id=element_id,
        type=element_type,
        points=pts,
        is_closed=is_closed,
        attrs=attrs or {},
        successor_ids=successor_ids,
        predecessor_ids=predecessor_ids,
        center=center,
        bound_radius=bound_radius,
    )


class NuscMap:
    """A parsed nuScenes map-expansion location, ready for ego-frame queries."""

    def __init__(self, elements: list[_GlobalElement]) -> None:
        self._elements = elements
        if elements:
            self._centers = np.stack([e.center for e in elements])
            self._radii = np.array([e.bound_radius for e in elements])
        else:
            self._centers = np.zeros((0, 2), dtype=np.float64)
            self._radii = np.zeros((0,), dtype=np.float64)

    @property
    def num_elements(self) -> int:
        return len(self._elements)

    # --- parsing -----------------------------------------------------------

    @classmethod
    def from_json(
        cls, path: str, lane_resolution_m: float = _LANE_RESOLUTION_M
    ) -> "NuscMap":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        node_xy = {n["token"]: (float(n["x"]), float(n["y"])) for n in data["node"]}
        line_by_token = {ln["token"]: ln for ln in data["line"]}
        polygon_by_token = {p["token"]: p for p in data["polygon"]}
        arcline = data.get("arcline_path_3", {})
        connectivity = data.get("connectivity", {})

        def line_coords(line_token: str) -> np.ndarray:
            tokens = line_by_token[line_token]["node_tokens"]
            return np.array([node_xy[t] for t in tokens], dtype=np.float64)

        def polygon_coords(polygon_token: str) -> np.ndarray:
            tokens = polygon_by_token[polygon_token]["exterior_node_tokens"]
            return np.array([node_xy[t] for t in tokens], dtype=np.float64)

        elements: list[_GlobalElement] = []

        def add(element: Optional[_GlobalElement]) -> None:
            if element is not None:
                elements.append(element)

        # --- lanes / lane connectors -> LANE_CENTER ------------------------
        for is_connector, layer in ((False, "lane"), (True, "lane_connector")):
            for record in data.get(layer, []):
                token = record["token"]
                paths = arcline.get(token)
                if not paths:
                    continue
                poses = discretize_lane(paths, lane_resolution_m)
                centerline = np.array([(p[0], p[1]) for p in poses], dtype=np.float64)
                conn = connectivity.get(token, {})
                attrs: dict[str, Any] = {"is_intersection": is_connector}
                lane_type = record.get("lane_type")
                if lane_type is not None:
                    attrs["lane_type"] = _LANE_TYPE.get(
                        lane_type, str(lane_type).lower()
                    )
                add(
                    _make_element(
                        f"lane_center_{token}",
                        MapElementType.LANE_CENTER,
                        centerline,
                        is_closed=False,
                        attrs=attrs,
                        successor_ids=tuple(conn.get("outgoing", [])),
                        predecessor_ids=tuple(conn.get("incoming", [])),
                    )
                )

        # --- lane_divider -> LANE_BOUNDARY (no paint sub-type in nuScenes) -
        for record in data.get("lane_divider", []):
            add(
                _make_element(
                    f"lane_boundary_{record['token']}",
                    MapElementType.LANE_BOUNDARY,
                    line_coords(record["line_token"]),
                    is_closed=False,
                    attrs={"paint_color": None, "paint_pattern": None},
                )
            )

        # --- road_divider -> ROAD_EDGE (median between traffic directions) -
        for record in data.get("road_divider", []):
            add(
                _make_element(
                    f"road_edge_{record['token']}",
                    MapElementType.ROAD_EDGE,
                    line_coords(record["line_token"]),
                    is_closed=False,
                    attrs={"road_edge_subtype": "median"},
                )
            )

        # --- polygon layers ------------------------------------------------
        for record in data.get("ped_crossing", []):
            add(
                _make_element(
                    f"crosswalk_{record['token']}",
                    MapElementType.CROSSWALK,
                    polygon_coords(record["polygon_token"]),
                    is_closed=True,
                )
            )
        for record in data.get("walkway", []):
            add(
                _make_element(
                    f"walkway_{record['token']}",
                    MapElementType.WALKWAY,
                    polygon_coords(record["polygon_token"]),
                    is_closed=True,
                )
            )
        for record in data.get("stop_line", []):
            stop_line_type = record.get("stop_line_type")
            add(
                _make_element(
                    f"stop_line_{record['token']}",
                    MapElementType.STOP_LINE,
                    polygon_coords(record["polygon_token"]),
                    is_closed=True,
                    attrs=(
                        {"stop_line_type": str(stop_line_type).lower()}
                        if stop_line_type
                        else {}
                    ),
                )
            )
        for record in data.get("drivable_area", []):
            for polygon_token in record.get("polygon_tokens", []):
                add(
                    _make_element(
                        f"drivable_{polygon_token}",
                        MapElementType.DRIVABLE_AREA,
                        polygon_coords(polygon_token),
                        is_closed=True,
                    )
                )
        for record in data.get("road_segment", []):
            if record.get("is_intersection"):
                add(
                    _make_element(
                        f"intersection_{record['token']}",
                        MapElementType.INTERSECTION,
                        polygon_coords(record["polygon_token"]),
                        is_closed=True,
                    )
                )

        return cls(elements)

    # --- per-frame ego query -----------------------------------------------

    def build_hd_map(
        self, pose_global_from_ego: np.ndarray, radius_m: float = _DEFAULT_RADIUS_M
    ) -> HDMap:
        """Elements within ``radius_m`` of the ego, transformed into the ego frame.

        Args:
            pose_global_from_ego: 4x4 ``T_global_from_ego`` (the ego pose).
            radius_m: ego-centric query radius in meters.
        """
        pose = np.asarray(pose_global_from_ego, dtype=np.float64)
        ego_xy = pose[:2, 3]
        ego_z = float(pose[2, 3])
        t_ego_from_global = np.linalg.inv(pose)

        if not self._elements:
            return HDMap(elements=[])
        # Cheap bounding-circle prefilter: keep elements whose circle meets the
        # ROI circle, then transform only those into the ego frame.
        distances = np.linalg.norm(self._centers - ego_xy, axis=1)
        in_roi = np.flatnonzero(distances <= (radius_m + self._radii))

        out: list[MapElement] = []
        for idx in in_roi:
            element = self._elements[idx]
            ego_xy_pts = _global_xy_to_ego(element.points, t_ego_from_global, ego_z)
            out.append(
                MapElement(
                    id=element.id,
                    type=element.type,
                    points=ego_xy_pts,
                    is_closed=element.is_closed,
                    successor_ids=list(element.successor_ids),
                    predecessor_ids=list(element.predecessor_ids),
                    attrs=dict(element.attrs),
                )
            )
        return HDMap(elements=out)


def _global_xy_to_ego(
    points_xy: np.ndarray, t_ego_from_global: np.ndarray, ego_z: float
) -> np.ndarray:
    """Transform ``(N, 2)`` global XY to ``(N, 2)`` float32 ego XY.

    The nuScenes vector map is 2D (z = 0). We lift each point to the ego's
    global altitude ``ego_z`` before applying the 4x4 inverse pose so that any
    ego pitch / roll does not couple a z-offset into the ego-frame XY (the same
    reasoning as the NAVSIM map handling); the result is a clean top-down slice.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    n = pts.shape[0]
    homog = np.column_stack(
        [pts, np.full(n, ego_z, dtype=np.float64), np.ones(n, dtype=np.float64)]
    )
    ego = (np.asarray(t_ego_from_global, dtype=np.float64) @ homog.T).T
    return ego[:, :2].astype(np.float32)
