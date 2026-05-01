import logging
from typing import Any

import numpy as np

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    Detections3DIdentityAdapter,
    HDMapIdentityAdapter,
    LidarPCIdentityAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.waymo_perception.hd_map_ego_crop import (
    WaymoHDMapEgoCropAggregator,
)
from standard_e2e.data_structures import (
    Detection3D,
    FrameDetections3D,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import CameraDirection, DetectionType
from standard_e2e.enums import TrajectoryComponent as TC

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame
from standard_e2e.utils import matrix_to_xyz_heading
from standard_e2e.utils.image_utils import (
    waymo_fetch_images_from_frame,
)
from standard_e2e.utils.waymo_lidar import frame_lasers_to_lidar_data

_logger = logging.getLogger(__name__)


class WaymoPerceptionDatasetProcessor(SourceDatasetProcessor):
    """Processor for the Waymo Perception dataset."""

    DATASET_NAME = "waymo_perception"
    CAMERAS_ORDER = {
        CameraDirection.FRONT: 1,
        CameraDirection.FRONT_LEFT: 2,
        CameraDirection.FRONT_RIGHT: 3,
        CameraDirection.SIDE_LEFT: 4,
        CameraDirection.SIDE_RIGHT: 5,
    }
    DEFAULT_HD_MAP_CROP_EXTENT_M = 75.0

    def __init__(
        self,
        common_output_path: str,
        split: str,
        source_data_path: str | None = None,
        hd_map_crop_extent_m: float | None = None,
        **kwargs: Any,
    ):
        """Construct the Waymo Perception processor.

        ``source_data_path`` (when provided) lets the default HD-map
        ego-crop aggregator find each segment's tfrecord (per ADR
        0007). When ``None`` the HD-map adapter and aggregator are
        omitted from defaults so unit tests / lidar-only conversions
        do not need the source data.
        """
        self._source_data_path = source_data_path
        self._hd_map_crop_extent_m = (
            hd_map_crop_extent_m
            if hd_map_crop_extent_m is not None
            else self.DEFAULT_HD_MAP_CROP_EXTENT_M
        )
        super().__init__(common_output_path=common_output_path, split=split, **kwargs)

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
        adapters: list[AbstractAdapter] = [
            PanoImageAdapter(),
            Detections3DIdentityAdapter(),
            LidarPCIdentityAdapter(),
        ]
        if self._source_data_path is not None:
            adapters.append(HDMapIdentityAdapter())
        return adapters

    def _get_default_context_aggregators(self):
        aggregators = [
            FuturePastStatesFromMatricesAggregator(self.output_path),
            FutureDetectionsAggregator(self.output_path),
        ]
        if self._source_data_path is not None:
            aggregators.append(
                WaymoHDMapEgoCropAggregator(
                    data_path=self.output_path,
                    source_data_path=self._source_data_path,
                    x_range=self._hd_map_crop_extent_m,
                    y_range=self._hd_map_crop_extent_m,
                )
            )
        return aggregators

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
        lidar_data = frame_lasers_to_lidar_data(frame) if frame.lasers else None
        extra_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        detections_3d = []
        # camera_synced_box good, but often empty
        # https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/label.proto#L108-#L133
        for laser_label in frame.laser_labels:
            box = laser_label.box
            if box.center_x == 0.0 and box.center_y == 0.0 and box.center_z == 0.0:
                # Real detections at the exact ego origin are implausible
                # (would be inside the ego sensor stack); treat (0,0,0) as a
                # malformed label and skip rather than emitting a ghost
                # Detection3D into the cache.
                _logger.warning(
                    "Skipping ghost laser label at ego origin "
                    "(segment=%s frame=%s agent=%s).",
                    segment_id,
                    frame_id,
                    laser_label.id,
                )
                continue
            detection = Detection3D(
                unique_agent_id=laser_label.id,
                detection_type=self._waymo_agent_type_to_canonical(laser_label.type),
                trajectory=Trajectory(
                    {
                        TC.TIMESTAMP: [timestamp],
                        TC.X: [box.center_x],
                        TC.Y: [box.center_y],
                        TC.Z: [box.center_z],
                        TC.HEADING: [box.heading],
                        TC.LENGTH: [box.length],
                        TC.WIDTH: [box.width],
                        TC.HEIGHT: [box.height],
                    }
                ),
            )
            detections_3d.append(detection)
        current_x, current_y, current_z, current_heading = matrix_to_xyz_heading(
            np.array(frame.pose.transform).reshape(4, 4)
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
            lidar=lidar_data,
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
