from typing import Optional

import numpy as np
import pandas as pd

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
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
from standard_e2e.third_party.waymo_open_dataset.utils import frame_utils
from standard_e2e.utils import matrix_to_xyz_heading
from standard_e2e.utils.image_utils import (
    waymo_fetch_images_from_frame,
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


class WaymoPerceptionDatasetProcessor(SourceDatasetProcessor):
    """Processor for the Waymo Perception dataset.

    HD-map handling note: Waymo's proto puts ``map_features`` only on the
    first frame of each segment. This processor caches the decoded map
    features per ``segment_id`` on first sighting and applies the
    per-frame inverse pose on subsequent frames. The cache is per-instance
    state, so HD-map preprocessing requires sequential processing
    (``do_parallel_processing=False`` on the converter); under
    multiprocessing, cache misses can occur on workers that didn't see
    the first frame.
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
        # Per-segment cache: segment_id -> list of (id, type, world_points, is_closed).
        self._segment_map_cache: dict[
            str, list[tuple[str, MapElementType, np.ndarray, bool]]
        ] = {}

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
        return [PanoImageAdapter(), LidarAdapter(), HDMapBEVAdapter()]

    def _get_default_context_aggregators(self):
        return [
            FuturePastStatesFromMatricesAggregator(self.output_path),
            FutureDetectionsAggregator(self.output_path),
        ]

    def _cache_map_features_for_segment(self, segment_id: str, map_features) -> None:
        """Decode Waymo ``map_features`` once per segment, keep world-frame xyz."""
        cached: list[tuple[str, MapElementType, np.ndarray, bool]] = []
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
            cached.append((str(feature.id), element_type, pts, is_closed))
        self._segment_map_cache[segment_id] = cached

    def _build_hd_map(
        self, segment_id: str, pose_world_from_vehicle: np.ndarray
    ) -> HDMap | None:
        """Apply inverse pose to the cached features and build vehicle-frame HDMap."""
        cached = self._segment_map_cache.get(segment_id)
        if cached is None:
            return None
        T_v_w = np.linalg.inv(pose_world_from_vehicle).astype(np.float32)
        elements: list[MapElement] = []
        for feat_id, element_type, world_pts, is_closed in cached:
            homog = np.concatenate(
                [world_pts, np.ones((len(world_pts), 1), dtype=np.float32)], axis=1
            )
            vehicle_pts = (homog @ T_v_w.T)[:, :3]
            elements.append(
                MapElement(
                    id=feat_id,
                    type=element_type,
                    points=vehicle_pts,
                    is_closed=is_closed,
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
        cameras_data = waymo_fetch_images_from_frame(frame)
        extra_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        detections_3d = []
        # camera_synced_box good, but often empty
        # https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/label.proto#L108-#L133
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
                detection_type=self._waymo_agent_type_to_canonical(laser_label.type),
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
        if len(frame.map_features) > 0 and segment_id not in self._segment_map_cache:
            self._cache_map_features_for_segment(segment_id, frame.map_features)
        hd_map = self._build_hd_map(segment_id, pose)

        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        points_per_laser, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )
        lidar_xyz = np.concatenate(points_per_laser, axis=0).astype(np.float32)
        lidar = LidarData(
            points=pd.DataFrame(lidar_xyz, columns=[c.value for c in LidarComponent])
        )
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
