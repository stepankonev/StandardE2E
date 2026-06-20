"""KITScenes Lanelet2 HD map -> unified ``HDMap``, parsed directly.

Each scene ships a Lanelet2 map as OSM XML (``maps/map.osm``) plus a WGS84
anchor (``maps/origin.json``). We can't depend on the ``lanelet2`` library (it
pins ``numpy<2`` against this project's numpy 2.x, like the nuScenes devkit), so
this module parses the OSM directly and translates each Lanelet2 primitive into
one or more :class:`~standard_e2e.data_structures.MapElement` instances,
following the cross-dataset taxonomy in ``docs/map_taxonomy.md``:

* ``lanelet`` road / emergency_lane / bicycle_lane -> ``LANE_CENTER``
  (centerline from the left/right bound midpoints, ``attrs.lane_type``)
* ``lanelet`` crosswalk -> ``CROSSWALK``; ``lanelet`` walkway -> ``WALKWAY``
  (closed polygon from the two bounds; ``walkway`` absent from the v1.0 sample)
* ``line_thin`` / ``line_thick`` / ``bike_marking`` -> ``LANE_BOUNDARY``
  (``attrs.paint_pattern``)
* ``curbstone`` / ``road_border`` -> ``ROAD_EDGE`` (``attrs.road_edge_subtype``)
* ``stop_line`` -> ``STOP_LINE``; ``traffic_light*`` -> ``TRAFFIC_LIGHT``

Not translated (no unified target, matching how the nuScenes processor skips
carparks / static signs): ``pole``, ``traffic_sign``, ``arrow``, ``symbol``,
``pedestrian_marking`` / ``zebra_marking`` ways (the crossing is captured by the
``crosswalk`` lanelet), ``virtual`` ways (used only as lanelet bounds when
computing centerlines), and ``regulatory_element`` relations (rules referencing
ways already emitted). Lane-graph connectivity (successor / predecessor /
neighbours) is not reconstructed in this release; the BEV rasterizer does not
use it.

Georeferencing: ``poses.txt`` is in the Lanelet2 map-local frame (UTM zone 32N
minus the ``origin.json`` anchor's UTM -- verified: the ego trajectory lies on
the map's node cloud), so map nodes projected with the same anchor share the ego
poses' frame and no extra GNSS reconciliation is needed. A :class:`KITScenesMap`
is parsed once per scene (cached by the processor) into map-local elements with a
bounding circle each; :meth:`KITScenesMap.build_hd_map` then ROI-filters and
transforms them into a single frame's ego coordinates.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np
from pyproj import Transformer

from standard_e2e.data_structures import HDMap, MapElement
from standard_e2e.enums import MapElementType

# All current KITScenes cities (Karlsruhe / Frankfurt / Sindelfingen) lie in UTM
# zone 32N; the devkit and ``origin.json`` projection both use EPSG:32632.
_UTM_ZONE_32N_EPSG = "EPSG:32632"
_WGS84_EPSG = "EPSG:4326"

# Default ego-centric query radius (m) for per-frame map extraction.
_DEFAULT_RADIUS_M = 64.0

# Lanelet2 lanelet subtype -> (unified MapElementType, attrs.lane_type). Subtypes
# absent here are skipped (logged by the caller is unnecessary -- they are simply
# not lane/walk geometry).
_LANELET_SUBTYPE_TO_TYPE: dict[str, tuple[MapElementType, Optional[str]]] = {
    "road": (MapElementType.LANE_CENTER, "vehicle"),
    "emergency_lane": (MapElementType.LANE_CENTER, "vehicle"),
    "bicycle_lane": (MapElementType.LANE_CENTER, "bike"),
    "crosswalk": (MapElementType.CROSSWALK, None),
    "walkway": (MapElementType.WALKWAY, None),
}

# Lanelet2 way type -> unified MapElementType for ways emitted standalone.
_WAY_TYPE_TO_TYPE: dict[str, MapElementType] = {
    "line_thin": MapElementType.LANE_BOUNDARY,
    "line_thick": MapElementType.LANE_BOUNDARY,
    "bike_marking": MapElementType.LANE_BOUNDARY,
    "curbstone": MapElementType.ROAD_EDGE,
    "road_border": MapElementType.ROAD_EDGE,
    "stop_line": MapElementType.STOP_LINE,
    "traffic_light": MapElementType.TRAFFIC_LIGHT,
    "traffic_light_pedestrians": MapElementType.TRAFFIC_LIGHT,
    "traffic_light_bikes": MapElementType.TRAFFIC_LIGHT,
    "traffic_light_misc": MapElementType.TRAFFIC_LIGHT,
}


@dataclass(frozen=True)
class _GlobalElement:
    """One map element in the map-local frame, with a bounding circle for cheap
    ROI prefiltering."""

    id: str
    type: MapElementType
    points: np.ndarray  # (N, 2) float64, map-local
    is_closed: bool
    attrs: dict[str, Any]
    center: np.ndarray  # (2,) float64
    bound_radius: float


def _resample_polyline(points: np.ndarray, num: int) -> np.ndarray:
    """Resample an ``(M, 2)`` polyline to ``(num, 2)`` evenly spaced by arc length."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return np.repeat(points[:1], num, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= 0.0:
        return np.repeat(points[:1], num, axis=0)
    targets = np.linspace(0.0, total, num)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1)


def _orient_to(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Reverse ``other`` if it runs opposite to ``reference`` (endpoint heuristic)."""
    same = np.linalg.norm(reference[0] - other[0]) + np.linalg.norm(
        reference[-1] - other[-1]
    )
    flipped = np.linalg.norm(reference[0] - other[-1]) + np.linalg.norm(
        reference[-1] - other[0]
    )
    return other[::-1] if flipped < same else other


def _lanelet_centerline(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Centerline as the midpoint of arc-length-resampled, co-oriented bounds."""
    right = _orient_to(left, right)
    num = max(2, left.shape[0], right.shape[0])
    center = 0.5 * (_resample_polyline(left, num) + _resample_polyline(right, num))
    return cast(np.ndarray, center)


def _lanelet_polygon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Closed polygon ring from a lanelet's two bounds (left + reversed right)."""
    right = _orient_to(left, right)
    return np.concatenate([left, right[::-1]], axis=0)


@dataclass
class _OsmData:
    """Parsed OSM geometry: node coords + way node-lists/tags + lanelet members."""

    node_xy: dict[str, np.ndarray]
    way_nodes: dict[str, list[str]]
    way_tags: dict[str, dict[str, str]]
    lanelets: list[dict[str, Any]] = field(default_factory=list)


def _tag_dict(element: ET.Element) -> dict[str, str]:
    return {
        tag.get("k", ""): tag.get("v", "")
        for tag in element.findall("tag")
        if tag.get("k") is not None
    }


def _parse_osm(
    osm_path: str, transformer: Transformer, origin_xy: np.ndarray
) -> _OsmData:
    """Parse the OSM XML into map-local node coords, way tables and lanelets."""
    root = ET.parse(osm_path).getroot()

    node_ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for node in root.findall("node"):
        node_id = node.get("id")
        lat = node.get("lat")
        lon = node.get("lon")
        if node_id is None or lat is None or lon is None:
            continue
        node_ids.append(node_id)
        lats.append(float(lat))
        lons.append(float(lon))
    easting, northing = transformer.transform(lons, lats)
    local = np.stack(
        [np.asarray(easting) - origin_xy[0], np.asarray(northing) - origin_xy[1]],
        axis=1,
    )
    node_xy = {node_id: local[i] for i, node_id in enumerate(node_ids)}

    way_nodes: dict[str, list[str]] = {}
    way_tags: dict[str, dict[str, str]] = {}
    for way in root.findall("way"):
        way_id = way.get("id")
        if way_id is None:
            continue
        way_nodes[way_id] = [
            nd.get("ref", "") for nd in way.findall("nd") if nd.get("ref")
        ]
        way_tags[way_id] = _tag_dict(way)

    lanelets: list[dict[str, Any]] = []
    for relation in root.findall("relation"):
        tags = _tag_dict(relation)
        if tags.get("type") != "lanelet":
            continue
        members = {
            member.get("role"): member.get("ref")
            for member in relation.findall("member")
            if member.get("role") in ("left", "right")
        }
        if "left" in members and "right" in members:
            lanelets.append(
                {
                    "id": relation.get("id"),
                    "subtype": tags.get("subtype", ""),
                    "left": members["left"],
                    "right": members["right"],
                }
            )

    return _OsmData(node_xy, way_nodes, way_tags, lanelets)


def _make_element(
    element_id: str,
    element_type: MapElementType,
    points: np.ndarray,
    is_closed: bool,
    attrs: Optional[dict[str, Any]] = None,
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
        center=center,
        bound_radius=bound_radius,
    )


class KITScenesMap:
    """A parsed KITScenes Lanelet2 map, ready for ego-frame queries."""

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
    def from_files(cls, osm_path: str, origin_path: str) -> "KITScenesMap":
        """Parse a scene's ``maps/map.osm`` + ``maps/origin.json``."""
        with open(origin_path, encoding="utf-8") as handle:
            origin = json.load(handle)
        transformer = Transformer.from_crs(
            _WGS84_EPSG, _UTM_ZONE_32N_EPSG, always_xy=True
        )
        origin_easting, origin_northing = transformer.transform(
            float(origin["longitude"]), float(origin["latitude"])
        )
        origin_xy = np.array([origin_easting, origin_northing], dtype=np.float64)

        osm = _parse_osm(osm_path, transformer, origin_xy)
        elements: list[_GlobalElement] = []

        def add(element: Optional[_GlobalElement]) -> None:
            if element is not None:
                elements.append(element)

        def way_coords(way_id: Optional[str]) -> Optional[np.ndarray]:
            if way_id is None or way_id not in osm.way_nodes:
                return None
            pts = [osm.node_xy[n] for n in osm.way_nodes[way_id] if n in osm.node_xy]
            return np.asarray(pts, dtype=np.float64) if pts else None

        # --- lanelets -> LANE_CENTER / CROSSWALK / WALKWAY -----------------
        for lanelet in osm.lanelets:
            mapping = _LANELET_SUBTYPE_TO_TYPE.get(lanelet["subtype"])
            if mapping is None:
                continue
            element_type, lane_type = mapping
            left = way_coords(lanelet["left"])
            right = way_coords(lanelet["right"])
            if left is None or right is None:
                continue
            attrs: dict[str, Any] = {"lanelet_subtype": lanelet["subtype"]}
            if element_type is MapElementType.LANE_CENTER:
                attrs["lane_type"] = lane_type
                add(
                    _make_element(
                        f"lane_center_{lanelet['id']}",
                        MapElementType.LANE_CENTER,
                        _lanelet_centerline(left, right),
                        is_closed=False,
                        attrs=attrs,
                    )
                )
            else:  # CROSSWALK / WALKWAY -> closed polygon from the two bounds
                add(
                    _make_element(
                        f"{element_type.value}_{lanelet['id']}",
                        element_type,
                        _lanelet_polygon(left, right),
                        is_closed=True,
                        attrs=attrs,
                    )
                )

        # --- standalone ways -> boundaries / edges / stop lines / lights ---
        for way_id, tags in osm.way_tags.items():
            way_type = _WAY_TYPE_TO_TYPE.get(tags.get("type", ""))
            if way_type is None:
                continue
            coords = way_coords(way_id)
            if coords is None:
                continue
            subtype = tags.get("subtype")
            way_attrs: dict[str, Any] = {}
            if way_type is MapElementType.LANE_BOUNDARY:
                way_attrs = {"paint_color": None, "paint_pattern": subtype}
                if tags.get("type") == "bike_marking":
                    way_attrs["lane_type"] = "bike"
            elif way_type is MapElementType.ROAD_EDGE:
                way_attrs = {
                    "road_edge_subtype": (
                        subtype if tags.get("type") == "curbstone" else "border"
                    )
                }
            elif way_type is MapElementType.TRAFFIC_LIGHT and subtype:
                way_attrs = {"tl_state_subtype_raw": subtype}
            add(
                _make_element(
                    f"{tags['type']}_{way_id}",
                    way_type,
                    coords,
                    is_closed=False,
                    attrs=way_attrs,
                )
            )

        return cls(elements)

    # --- per-frame ego query -----------------------------------------------

    def build_hd_map(
        self, pose_maplocal_from_ego: np.ndarray, radius_m: float = _DEFAULT_RADIUS_M
    ) -> HDMap:
        """Elements within ``radius_m`` of the ego, transformed into the ego frame.

        Args:
            pose_maplocal_from_ego: 4x4 ``T_maplocal_from_ego`` (the ego pose).
            radius_m: ego-centric query radius in meters.
        """
        pose = np.asarray(pose_maplocal_from_ego, dtype=np.float64)
        ego_xy = pose[:2, 3]
        ego_z = float(pose[2, 3])
        t_ego_from_map = np.linalg.inv(pose)

        if not self._elements:
            return HDMap(elements=[])
        distances = np.linalg.norm(self._centers - ego_xy, axis=1)
        in_roi = np.flatnonzero(distances <= (radius_m + self._radii))

        out: list[MapElement] = []
        for idx in in_roi:
            element = self._elements[idx]
            ego_xy_pts = _maplocal_xy_to_ego(element.points, t_ego_from_map, ego_z)
            out.append(
                MapElement(
                    id=element.id,
                    type=element.type,
                    points=ego_xy_pts,
                    is_closed=element.is_closed,
                    attrs=dict(element.attrs),
                )
            )
        return HDMap(elements=out)


def _maplocal_xy_to_ego(
    points_xy: np.ndarray, t_ego_from_map: np.ndarray, ego_z: float
) -> np.ndarray:
    """Transform ``(N, 2)`` map-local XY to ``(N, 2)`` float32 ego XY.

    The Lanelet2 map is treated as 2D; each point is lifted to the ego's
    map-local altitude ``ego_z`` before applying the 4x4 inverse pose so that any
    ego pitch / roll does not couple a z-offset into the ego-frame XY (the same
    reasoning as the nuScenes / NAVSIM map handling); the result is a clean
    top-down slice.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    n = pts.shape[0]
    homogeneous = np.column_stack(
        [pts, np.full(n, ego_z, dtype=np.float64), np.ones(n, dtype=np.float64)]
    )
    ego = (np.asarray(t_ego_from_map, dtype=np.float64) @ homogeneous.T).T
    return ego[:, :2].astype(np.float32)
