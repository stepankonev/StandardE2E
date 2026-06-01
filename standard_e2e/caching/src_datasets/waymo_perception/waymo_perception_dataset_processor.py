import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    HDMap,
    LidarData,
    MapElement,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    MapElementType,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame
from standard_e2e.utils import matrix_to_xyz_heading
from standard_e2e.utils.image_utils import (
    waymo_fetch_images_from_frame,
)
from standard_e2e.utils.waymo_lidar_numpy import (
    numpy_convert_range_image_to_point_cloud,
    numpy_parse_range_image_and_camera_projection,
)

# (waymo_field_name, MapElementType, polyline_attr_or_None_for_point, is_closed)
_MAP_FEATURE_KIND: dict[str, tuple[MapElementType, Optional[str], bool]] = {
    "lane": (MapElementType.LANE_CENTER, "polyline", False),
    "road_edge": (MapElementType.ROAD_EDGE, "polyline", False),
    "road_line": (MapElementType.LANE_BOUNDARY, "polyline", False),
    "stop_sign": (MapElementType.STOP_SIGN, None, False),  # single position point
    "crosswalk": (MapElementType.CROSSWALK, "polygon", True),
    "speed_bump": (MapElementType.SPEED_BUMP, "polygon", True),
    "driveway": (MapElementType.DRIVEWAY, "polygon", True),
}

# Waymo LaneType enum -> our normalised lane_type string.
# Source: waymo_open_dataset/protos/map.proto LaneCenter.LaneType.
# 0 (UNDEFINED) maps to absent.
_WAYMO_LANE_TYPE_TO_NORMALISED: dict[int, str] = {
    1: "freeway",
    2: "surface_street",
    3: "bike",
}

# Waymo RoadLineType -> (paint_color, paint_pattern) and raw enum name.
# See docs/lane_paint_comparison.md for the full vocabulary mapping.
_WAYMO_ROAD_LINE_TYPE_TO_PAINT: dict[int, tuple[str, str]] = {
    1: ("white", "dashed"),  # TYPE_BROKEN_SINGLE_WHITE
    2: ("white", "solid"),  # TYPE_SOLID_SINGLE_WHITE
    3: ("white", "double_solid"),  # TYPE_SOLID_DOUBLE_WHITE
    4: ("yellow", "dashed"),  # TYPE_BROKEN_SINGLE_YELLOW
    5: ("yellow", "double_dashed"),  # TYPE_BROKEN_DOUBLE_YELLOW
    6: ("yellow", "solid"),  # TYPE_SOLID_SINGLE_YELLOW
    7: ("yellow", "double_solid"),  # TYPE_SOLID_DOUBLE_YELLOW
    8: ("yellow", "solid_dashed"),  # TYPE_PASSING_DOUBLE_YELLOW
}
_WAYMO_ROAD_LINE_TYPE_NAMES: dict[int, str] = {
    0: "TYPE_UNKNOWN",
    1: "BROKEN_SINGLE_WHITE",
    2: "SOLID_SINGLE_WHITE",
    3: "SOLID_DOUBLE_WHITE",
    4: "BROKEN_SINGLE_YELLOW",
    5: "BROKEN_DOUBLE_YELLOW",
    6: "SOLID_SINGLE_YELLOW",
    7: "SOLID_DOUBLE_YELLOW",
    8: "PASSING_DOUBLE_YELLOW",
}

# Waymo RoadEdgeType -> normalised road_edge_subtype.
# 0 (UNKNOWN) maps to absent.
_WAYMO_ROAD_EDGE_TYPE_TO_SUBTYPE: dict[int, str] = {
    1: "boundary",  # ROAD_EDGE_BOUNDARY (curb)
    2: "median",  # ROAD_EDGE_MEDIAN
}


class WaymoPerceptionDatasetProcessor(SourceDatasetProcessor):
    """Processor for the Waymo Perception dataset.

    HD-map handling: Waymo's proto puts ``map_features`` only on the
    first frame of each segment. This processor caches the decoded
    world-frame map features per ``segment_id`` and applies the per-frame
    inverse pose on every frame. :class:`WaymoPerceptionDatasetConverter`
    pre-populates the cache before parallel work begins, so both
    sequential and parallel preprocessing produce HD maps for every frame.
    """

    DATASET_NAME = "waymo_perception"
    CAMERAS_ORDER = {
        CameraDirection.FRONT: 1,
        CameraDirection.FRONT_LEFT: 2,
        CameraDirection.FRONT_RIGHT: 3,
        CameraDirection.SIDE_LEFT: 4,
        CameraDirection.SIDE_RIGHT: 5,
    }

    def __init__(
        self,
        common_output_path: str,
        split: str,
        index_data_generator: IndexDataGenerator | None = None,
        adapters: list[AbstractAdapter] | None = None,
        context_aggregators: list[SegmentContextAggregator] | None = None,
    ):
        super().__init__(
            common_output_path=common_output_path,
            split=split,
            index_data_generator=index_data_generator,
            adapters=adapters,
            context_aggregators=context_aggregators,
        )
        # Per-segment cache: segment_id -> list of dicts carrying the
        # decoded world-frame points + per-feature attrs and lane-graph
        # links. Keys: id, type, points_world, is_closed,
        # successor_ids, predecessor_ids, left_neighbor_id,
        # right_neighbor_id, attrs.
        self._segment_map_cache: dict[str, list[dict[str, Any]]] = {}
        # When set, prescanned segment data is spilled to disk as
        # ``<dir>/<segment_id>.pkl``. The in-memory dict is then kept
        # tiny (each worker lazily loads only the segments it touches).
        # This is the path that lets the Pool initializer avoid shipping
        # the entire ~200 MB cache to every worker — workers receive the
        # processor with an empty in-memory dict + a path string instead.
        self._segment_cache_dir: Optional[str] = None

    @property
    def allowed_splits(self) -> list[str]:
        """Return the list of allowed splits for the Waymo Perception dataset."""
        return [
            "training",
            "validation",
            "testing",
            "testing_3d_camera_only_detection",
            "domain_adaptation",
        ]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        """Get the adapters for the Waymo Perception dataset."""
        return [
            PanoImageAdapter(),
            LidarAdapter(),
            HDMapBEVAdapter(),
            Detections3DIdentityAdapter(),
        ]

    def _get_default_context_aggregators(self):
        return [
            FuturePastStatesFromMatricesAggregator(self.output_path),
            FutureDetectionsAggregator(self.output_path),
        ]

    def _cache_map_features_for_segment(self, segment_id: str, map_features) -> None:
        """Decode Waymo ``map_features`` once per segment, keep world-frame xyz.

        See docs/map_element_translation.md §5.1 for the per-feature
        translation rules. Per-feature attrs (lane_type, paint_color,
        paint_pattern, paint_subtype_raw, road_edge_subtype,
        speed_limit_mph, is_intersection, controlled_lane_id) are
        rebucketed at cache time from the source proto enums.
        """
        cached: list[dict[str, Any]] = []
        for feature in map_features:
            kind = feature.WhichOneof("feature_data")
            mapping = _MAP_FEATURE_KIND.get(kind)
            if mapping is None:
                continue
            element_type, polyline_attr, is_closed = mapping
            sub = getattr(feature, kind)
            if polyline_attr is None:  # stop_sign single position
                p = sub.position
                pts = np.array([[p.x, p.y, p.z]], dtype=np.float32)
            else:
                points_proto = getattr(sub, polyline_attr)
                pts = np.array(
                    [[p.x, p.y, p.z] for p in points_proto], dtype=np.float32
                )

            attrs: dict[str, Any] = {}
            successor_ids: list[str] = []
            predecessor_ids: list[str] = []
            left_neighbor_id: Optional[str] = None
            right_neighbor_id: Optional[str] = None

            if kind == "lane":
                lane_type_norm = _WAYMO_LANE_TYPE_TO_NORMALISED.get(int(sub.type))
                if lane_type_norm is not None:
                    attrs["lane_type"] = lane_type_norm
                attrs["is_intersection"] = bool(sub.interpolating)
                # Waymo's proto stores 0.0 for missing speed limits — only
                # surface a value when explicitly populated.
                if sub.speed_limit_mph > 0:
                    attrs["speed_limit_mph"] = float(sub.speed_limit_mph)
                successor_ids = [str(i) for i in sub.exit_lanes]
                predecessor_ids = [str(i) for i in sub.entry_lanes]
                if len(sub.left_neighbors) > 0:
                    left_neighbor_id = str(sub.left_neighbors[0].feature_id)
                if len(sub.right_neighbors) > 0:
                    right_neighbor_id = str(sub.right_neighbors[0].feature_id)
            elif kind == "road_line":
                paint = _WAYMO_ROAD_LINE_TYPE_TO_PAINT.get(int(sub.type))
                if paint is not None:
                    attrs["paint_color"], attrs["paint_pattern"] = paint
                attrs["paint_subtype_raw"] = _WAYMO_ROAD_LINE_TYPE_NAMES.get(
                    int(sub.type), str(int(sub.type))
                )
            elif kind == "road_edge":
                subtype = _WAYMO_ROAD_EDGE_TYPE_TO_SUBTYPE.get(int(sub.type))
                if subtype is not None:
                    attrs["road_edge_subtype"] = subtype
            elif kind == "stop_sign":
                if len(sub.lane) > 0:
                    attrs["controlled_lane_id"] = str(sub.lane[0])

            cached.append(
                {
                    "id": str(feature.id),
                    "type": element_type,
                    "points_world": pts,
                    "is_closed": is_closed,
                    "successor_ids": successor_ids,
                    "predecessor_ids": predecessor_ids,
                    "left_neighbor_id": left_neighbor_id,
                    "right_neighbor_id": right_neighbor_id,
                    "attrs": attrs,
                }
            )
        if self._segment_cache_dir is not None:
            # Disk-spill mode: write the per-segment cache to a file and
            # drop the in-memory copy. Workers will lazy-load via
            # ``_build_hd_map`` on first hit for the segment.
            os.makedirs(self._segment_cache_dir, exist_ok=True)
            tmp = os.path.join(self._segment_cache_dir, f"{segment_id}.pkl.tmp")
            final = os.path.join(self._segment_cache_dir, f"{segment_id}.pkl")
            with open(tmp, "wb") as fp:
                pickle.dump(cached, fp, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, final)
        else:
            self._segment_map_cache[segment_id] = cached

    def _build_hd_map(
        self, segment_id: str, pose_world_from_vehicle: np.ndarray
    ) -> HDMap | None:
        """Apply inverse pose to the cached features and build vehicle-frame HDMap."""
        cached = self._segment_map_cache.get(segment_id)
        if cached is None and self._segment_cache_dir is not None:
            # Disk-spill mode: lazy-load this segment's cache from disk
            # on first hit, then keep it in memory for subsequent frames
            # of the same segment.
            path = os.path.join(self._segment_cache_dir, f"{segment_id}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fp:
                    cached = pickle.load(fp)
                self._segment_map_cache[segment_id] = cached
        if cached is None:
            return None
        T_v_w = np.linalg.inv(pose_world_from_vehicle).astype(np.float32)
        elements: list[MapElement] = []
        for entry in cached:
            world_pts = entry["points_world"]
            homog = np.concatenate(
                [world_pts, np.ones((len(world_pts), 1), dtype=np.float32)], axis=1
            )
            vehicle_pts = (homog @ T_v_w.T)[:, :3]
            elements.append(
                MapElement(
                    id=entry["id"],
                    type=entry["type"],
                    points=vehicle_pts,
                    is_closed=entry["is_closed"],
                    successor_ids=entry["successor_ids"],
                    predecessor_ids=entry["predecessor_ids"],
                    left_neighbor_id=entry["left_neighbor_id"],
                    right_neighbor_id=entry["right_neighbor_id"],
                    attrs=entry["attrs"],
                )
            )
        return HDMap(elements=elements)

    def _waymo_agent_type_to_canonical(self, agent_type):
        if agent_type == 0:
            return DetectionType.UNKNOWN
        elif agent_type == 1:
            return DetectionType.VEHICLE
        elif agent_type == 2:
            return DetectionType.PEDESTRIAN
        elif agent_type == 3:
            return DetectionType.SIGN
        elif agent_type == 4:
            return DetectionType.BICYCLE
        else:
            return DetectionType.UNKNOWN

    def _prepare_standardized_frame_data(self, raw_frame_data) -> StandardFrameData:
        frame = WaymoFrame()
        frame.ParseFromString(raw_frame_data.numpy())
        segment_id = frame.context.name
        timestamp = frame.timestamp_micros / 1_000_000.0
        frame_id = int(frame.timestamp_micros)
        cameras_data = (
            waymo_fetch_images_from_frame(frame) if self.needs_attr("cameras") else {}
        )
        extra_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        detections_3d = []
        # camera_synced_box good, but often empty
        # https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/label.proto#L108-#L133
        if self.needs_attr("frame_detections_3d"):
            for laser_label in frame.laser_labels:
                if np.allclose(
                    np.array(
                        [
                            laser_label.box.center_x,
                            laser_label.box.center_y,
                            laser_label.box.center_z,
                        ]
                    ),
                    np.zeros((3,)),
                ):
                    print("Laser label is centered at the origin.")
                gt_box = laser_label.box
                detection = Detection3D(
                    unique_agent_id=laser_label.id,
                    detection_type=self._waymo_agent_type_to_canonical(
                        laser_label.type
                    ),
                    trajectory=Trajectory(
                        {
                            TC.TIMESTAMP: [timestamp],
                            TC.X: [gt_box.center_x],
                            TC.Y: [gt_box.center_y],
                            TC.Z: [gt_box.center_z],
                            TC.HEADING: [gt_box.heading],
                            TC.LENGTH: [gt_box.length],
                            TC.WIDTH: [gt_box.width],
                            TC.HEIGHT: [gt_box.height],
                        }
                    ),
                )
                detections_3d.append(detection)
        current_x, current_y, current_z, current_heading = matrix_to_xyz_heading(
            np.array(frame.pose.transform).reshape(4, 4)
        )

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        if (
            self.needs_attr("hd_map")
            and len(frame.map_features) > 0
            and segment_id not in self._segment_map_cache
        ):
            self._cache_map_features_for_segment(segment_id, frame.map_features)
        hd_map = (
            self._build_hd_map(segment_id, pose) if self.needs_attr("hd_map") else None
        )

        if self.needs_attr("lidar"):
            # Pure-numpy decode (see ``standard_e2e/utils/waymo_lidar_numpy.py``)
            # — avoids TF runtime overhead in the worker hot path, which
            # at par-32 dominated per-frame wall time.
            range_images, _, _, range_image_top_pose = (
                numpy_parse_range_image_and_camera_projection(frame)
            )
            points_per_laser = numpy_convert_range_image_to_point_cloud(
                frame, range_images, range_image_top_pose
            )
            lidar_xyz = np.concatenate(points_per_laser, axis=0).astype(np.float32)
            lidar: Optional[LidarData] = LidarData(
                points=pd.DataFrame(
                    lidar_xyz, columns=[c.value for c in LidarComponent]
                )
            )
        else:
            lidar = None
        frame_data = StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=segment_id,
            frame_id=frame_id,
            timestamp=timestamp,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [timestamp],
                    TC.X: [current_x],
                    TC.Y: [current_y],
                    TC.Z: [current_z],
                    TC.HEADING: [current_heading],
                }
            ),
            split=self._split,
            cameras=cameras_data,
            lidar=lidar,
            hd_map=hd_map,
            frame_detections_3d=FrameDetections3D(detections=detections_3d),
            aux_data={
                **extra_data,
                "pose_matrix": np.array(frame.pose.transform).reshape(4, 4),
            },
            extra_index_data=extra_data,
        )
        return frame_data

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        return self.DATASET_NAME
