"""Source-dataset processor for the Argoverse 2 (AV2) lidar dataset.

The AV2 lidar split shares the on-disk format of the sensor split — same
``city_SE3_egovehicle.feather``, same ``map/log_map_archive_*.json``, same
``sensors/lidar/<timestamp_ns>.feather`` sweep layout — minus the
``sensors/cameras/`` directory and the ``annotations.feather`` file. We
inherit from :class:`Av2SensorDatasetProcessor` and override the few
hooks that touch cameras or 3D box annotations so the rest of the
behaviour (per-log cache, ego-pose handling, lidar loading, HD-map
construction) is shared verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from av2.map.map_api import ArgoverseStaticMap

from standard_e2e.caching.adapters import (
    AbstractAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.av2_sensor.av2_sensor_dataset_processor import (
    Av2SensorDatasetProcessor,
    _se3_from_quat_translation,
)
from standard_e2e.data_structures import CameraData, Detection3D
from standard_e2e.enums import CameraDirection


class Av2LidarDatasetProcessor(Av2SensorDatasetProcessor):
    """Processor for the Argoverse 2 lidar dataset.

    Modality coverage:

    - ✓ lidar (per-sweep merged point cloud, ego frame)
    - ✓ HD map (vector lanes, drivable area, crosswalks; same taxonomy
      as :class:`Av2SensorDatasetProcessor`)
    - ✗ cameras (AV2 lidar logs have no camera images)
    - ✗ 3D detections (AV2 lidar logs have no ``annotations.feather``)

    The two missing modalities are surfaced as defaults at training time
    via :class:`~standard_e2e.dataset_utils.modality_defaults.ModalityDefaults`,
    so a model trained on AV2 lidar can share batches with cameras+detections
    datasets like AV2 sensor or Waymo Perception.
    """

    DATASET_NAME = "av2_lidar"

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [LidarAdapter(), HDMapBEVAdapter()]

    def _get_default_context_aggregators(self):
        # No FutureDetectionsAggregator — there are no detections to aggregate.
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    def _refresh_log_cache(self, log_dir: Path) -> None:
        """Lidar-only per-log cache: ego-pose table + map; no cameras, no
        annotations."""
        if self._cached_log_dir == log_dir:
            return
        logging.info("Loading per-log AV2 lidar state for %s", log_dir.name)

        # Camera + annotation caches stay empty — the corresponding
        # ``_build_*`` overrides below short-circuit and never read them.
        self._camera_intrinsics.clear()
        self._camera_distortion.clear()
        self._camera_extrinsics.clear()
        self._camera_image_paths.clear()
        self._annotations_by_ts = {}

        ego_df = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
        self._ego_pose_by_ts = {}
        for raw_row in ego_df.itertuples(index=False):
            row = cast(Any, raw_row)
            self._ego_pose_by_ts[int(row.timestamp_ns)] = _se3_from_quat_translation(
                row.qw, row.qx, row.qy, row.qz, row.tx_m, row.ty_m, row.tz_m
            )

        self._map = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=False)
        self._cached_log_dir = log_dir

    def _build_camera_dict(
        self, log_dir: Path, sweep_ts_ns: int
    ) -> dict[CameraDirection, CameraData]:
        return {}

    def _build_detections(
        self, sweep_ts_ns: int, timestamp_s: float
    ) -> list[Detection3D]:
        return []
