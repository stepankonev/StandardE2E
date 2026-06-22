"""Source-dataset processor for KITScenes Multimodal.

This integration targets the **Multimodal** variant -- the StandardE2E dataset
key is ``kitscenes_multimodal`` (the planned KITScenes-LongTail variant will be
added as a separate sibling). KITScenes Multimodal (KIT / MRT, arXiv 2606.02956)
is a large European urban dataset whose distinctive feature is a dense,
georeferenced **Lanelet2 HD map** alongside a rich sensor suite (9 cameras, 7
LiDARs, 3 imaging radars, dual GNSS/INS) captured at 10 Hz over ~1000 scenes. The
release ships per-split tarballs on Hugging Face and must be extracted first (see
``scripts/prepare_dataset_kitscenes_multimodal.sh`` / :mod:`._kitscenes_io`).

Mapping to ``StandardFrameData`` (this integration covers the surround cameras,
the top LiDAR, the ego trajectory and the HD map):

* One **frame = one synchronized 10 Hz snapshot** (a 10-digit frame index); one
  **segment = one scene** (a UUID directory). The reference timeline
  (``timestamp.reference.txt``) gives per-frame timestamps and ``poses.txt`` the
  per-frame ego pose, both 1:1 with the frame index.
* **cameras**: the six surround ``camera_ring_*`` views, each keyed by the
  canonical :class:`CameraDirection` matching its facing. Intrinsics are the
  pinhole ``K``; extrinsics are ``T_ego_from_camera`` (the calib's
  ``T_to_reference``, since the reference ``base_frame`` is the ego frame).
* **lidar**: the 128-beam ``lidar_top`` xyz, de-discretized and moved into the
  ego frame via its extrinsic (the identity for KITScenes -- ``lidar_top`` is the
  ego origin -- applied for generality).
* **hd_map**: the per-scene Lanelet2 map (``maps/map.osm``), translated into the
  unified taxonomy and cropped to an ego-centric ROI per frame (see
  :mod:`._kitscenes_map`). Consumed by ``HDMapBEVAdapter``.
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose
  ``T_maplocal_from_ego`` in the Lanelet2 map-local frame (``poses.txt``).
  Past/future ego trajectories are produced by
  :class:`FuturePastStatesFromMatricesAggregator`, as for AV2 / Waymo / VoD /
  TruckDrive.

Ego frame: ``base_frame`` (FLU -- x-forward, y-left, z-up) with origin at
``lidar_top``; the LiDAR points and the map are both expressed in / relative to
it, so points, map and ego pose are mutually consistent.

Not ingested (no StandardE2E target in this release): the three "base" cameras
(high-resolution front-center + rectified stereo pair), the other six LiDARs, the
three imaging radars, the GNSS/INS streams and the ``processed/`` model outputs.
KITScenes ships **no 3D object boxes / tracks** (its annotation product is the HD
map), so every frame carries an empty ``frame_detections_3d``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    CamerasIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal import _kitscenes_io as io
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_geometry import (
    RingCameraCalibration,
    parse_calibration,
    pose_from_tum,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_map import (
    KITScenesMap,
)
from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_splits import (
    ALLOWED_SPLITS,
)
from standard_e2e.data_structures import (
    CameraData,
    FrameDetections3D,
    LidarData,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import CameraDirection, LidarComponent, StandardFrameDataField
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import matrix_to_xyz_heading, transform_points


class KITScenesMultimodalDatasetProcessor(SourceDatasetProcessor):
    """Processor for the KITScenes Multimodal dataset.

    Per-scene state (calibration, ego-pose table, reference timestamps and the
    parsed HD map) is read once and reused across every frame of the scene; the
    cache is keyed by ``scene_id`` so pooled workers that hop between scenes only
    reload on transition. The HD map is parsed only when an adapter consumes it.
    """

    DATASET_NAME = "kitscenes_multimodal"

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
        # Per-worker, per-scene cache (populated lazily in the worker).
        self._cached_scene_id: Optional[str] = None
        self._camera_calibs: dict[CameraDirection, RingCameraCalibration] = {}
        self._lidar_top_extrinsic: np.ndarray = np.eye(4, dtype=np.float64)
        self._poses: np.ndarray = np.zeros((0, 8), dtype=np.float64)
        self._reference_timestamps: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._map: Optional[KITScenesMap] = None

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        return list(ALLOWED_SPLITS)

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            CamerasIdentityAdapter(),
            LidarAdapter(),
            HDMapBEVAdapter(),
        ]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- per-scene cache ---------------------------------------------------

    def _refresh_scene_cache(self, ref: io.SceneRef) -> None:
        if self._cached_scene_id == ref.scene_id:
            return
        calib = io.read_calibration(ref.calib_path)
        self._camera_calibs, self._lidar_top_extrinsic = parse_calibration(calib)
        self._poses = io.read_poses(ref.poses_path)
        self._reference_timestamps = io.read_reference_timestamps(
            ref.reference_timestamps_path
        )

        self._map = None
        if self.needs_attr(StandardFrameDataField.HD_MAP):
            try:
                self._map = KITScenesMap.from_files(
                    ref.map_osm_path, ref.map_origin_path
                )
            except FileNotFoundError:
                logging.warning(
                    "KITScenes: no HD map for scene %s (%s); hd_map will be empty.",
                    ref.scene_id,
                    ref.map_osm_path,
                )

        self._cached_scene_id = ref.scene_id
        logging.debug(
            "KITScenes scene %s cached: %d poses, %d cameras, map=%s",
            ref.scene_id,
            self._poses.shape[0],
            len(self._camera_calibs),
            "yes" if self._map is not None else "no",
        )

    # --- modality builders -------------------------------------------------

    def _build_cameras(
        self, ref: io.SceneRef, frame_index: int
    ) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for direction, calib in self._camera_calibs.items():
            path = ref.camera_path(calib.camera_name, frame_index)
            cameras[direction] = CameraData(
                camera_direction=direction,
                image=io.read_image(path),
                intrinsics=calib.intrinsics,
                extrinsics=calib.extrinsics.astype(np.float32),
                is_fisheye=False,
            )
        return cameras

    def _build_lidar(self, ref: io.SceneRef, frame_index: int) -> Optional[LidarData]:
        xyz_sensor = io.read_lidar_top_xyz(ref.lidar_top_path(frame_index))
        xyz_ego = transform_points(self._lidar_top_extrinsic, xyz_sensor)
        return LidarData(
            points=pd.DataFrame(xyz_ego, columns=[c.value for c in LidarComponent])
        )

    # --- main entry point --------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        ref, frame_index = raw_frame_data
        self._refresh_scene_cache(ref)
        if frame_index >= self._poses.shape[0] or frame_index >= len(
            self._reference_timestamps
        ):
            raise KeyError(
                f"frame {frame_index} out of range for scene {ref.scene_id} "
                f"({self._poses.shape[0]} poses, "
                f"{len(self._reference_timestamps)} timestamps)"
            )

        pose_maplocal_from_ego = pose_from_tum(self._poses[frame_index])
        timestamp_s = float(self._reference_timestamps[frame_index])
        x, y, z, heading = matrix_to_xyz_heading(pose_maplocal_from_ego)

        cameras = (
            self._build_cameras(ref, frame_index)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(ref, frame_index)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )
        hd_map = (
            self._map.build_hd_map(pose_maplocal_from_ego)
            if self._map is not None and self.needs_attr(StandardFrameDataField.HD_MAP)
            else None
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.scene_id,
            frame_id=frame_index,
            timestamp=timestamp_s,
            split=self._split,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [timestamp_s],
                    TC.X: [x],
                    TC.Y: [y],
                    TC.Z: [z],
                    TC.HEADING: [heading],
                }
            ),
            cameras=cameras,
            lidar=lidar,
            hd_map=hd_map,
            # KITScenes ships no 3D object boxes; detections are always empty.
            frame_detections_3d=FrameDetections3D(detections=[]),
            aux_data={"pose_matrix": pose_maplocal_from_ego.astype(np.float64)},
            extra_index_data={"scene": ref.scene_id, "frame_index": frame_index},
        )
