"""Source-dataset processor for nuScenes (v1.0).

nuScenes (Motional, CVPR 2020) is the de-facto end-to-end / BEV benchmark: 1000
~20 s scenes from a 6-camera surround rig, a 32-beam ``LIDAR_TOP``, 5 radars and
densely annotated 3D boxes at 2 Hz keyframes. StandardE2E reads the JSON tables
directly (see :mod:`._nuscenes_io`; the ``nuscenes-devkit`` is reference-only --
it pins ``numpy<2`` against this project's numpy 2.x).

Mapping to ``StandardFrameData`` (this release covers cameras, the ego
trajectory, ``LIDAR_TOP``, 3D detections and -- when the separate map-expansion
pack is present -- the vector HD map; radar has no target yet):

* One **frame = one sample** (2 Hz keyframe); one **scene = one cache segment**.
* **cameras**: the 6 surround cameras, keyed by the canonical
  :class:`CameraDirection` matching each ``CAM_*`` channel. Intrinsics are the
  pinhole ``camera_intrinsic``; extrinsics are ``T_ego_from_camera``.
* **lidar**: the ``LIDAR_TOP`` xyz, moved from the sensor frame into the ego
  frame with ``T_ego_from_lidar``.
* **frame_detections_3d**: the ``sample_annotation`` boxes, transformed from the
  global frame into the ego frame. nuScenes does not annotate the ego vehicle, so
  nothing is excluded; the per-class ``category_name`` is folded into the coarse
  :class:`DetectionType` taxonomy.
* **hd_map**: the vector map-expansion within an ego-centric ROI, translated to
  the unified :class:`~standard_e2e.enums.MapElementType` taxonomy (lane centers,
  dividers, crossings, walkways, stop lines, drivable area, intersections) in the
  ego frame; consumed by :class:`HDMapBEVAdapter`. Built only when the
  map-expansion pack is unzipped into ``<dataroot>/maps/`` (see
  :mod:`._nuscenes_map`).
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose
  ``T_global_from_ego`` (the sample's ``LIDAR_TOP`` ego_pose). Past/future ego
  trajectories come from :class:`FuturePastStatesFromMatricesAggregator`, exactly
  as for AV2 / Waymo / NAVSIM / TruckDrive.

Ego frame: nuScenes' ego (IMU) frame at the lidar capture time -- the frame the
calibrated-sensor extrinsics are expressed against. LiDAR is moved into it and
boxes are transformed into it, so points and boxes are mutually consistent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    HDMapBEVAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.nuscenes import _nuscenes_io as io
from standard_e2e.caching.src_datasets.nuscenes._nuscenes_map import NuscMap
from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
    HDMap,
    LidarData,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    StandardFrameDataField,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import (
    matrix_to_xyz_heading,
    quat_wxyz_to_rotmat,
    se3,
    transform_points,
)

# The 6 nuScenes cameras map exactly onto the canonical surround directions.
_CHANNEL_TO_DIRECTION: dict[str, CameraDirection] = {
    "CAM_FRONT": CameraDirection.FRONT,
    "CAM_FRONT_LEFT": CameraDirection.FRONT_LEFT,
    "CAM_FRONT_RIGHT": CameraDirection.FRONT_RIGHT,
    "CAM_BACK": CameraDirection.REAR,
    "CAM_BACK_LEFT": CameraDirection.REAR_LEFT,
    "CAM_BACK_RIGHT": CameraDirection.REAR_RIGHT,
}


def detection_type_for_category(category_name: str) -> DetectionType:
    """Fold a nuScenes ``category_name`` into the coarse ``DetectionType``.

    nuScenes uses a dot-separated taxonomy (frozen across releases), so prefix
    rules cover every category:

    * ``vehicle.bicycle`` / ``vehicle.motorcycle`` -> ``BICYCLE`` (two-wheelers,
      matching TruckDrive's Two-Wheel bucket);
    * any other ``vehicle.*`` (car, truck, bus, trailer, construction,
      emergency) -> ``VEHICLE``;
    * ``human.pedestrian.*`` and ``animal`` -> ``PEDESTRIAN`` (the
      vulnerable-road-user bucket, matching TruckDrive's Person grouping);
    * ``movable_object.*`` / ``static_object.*`` (barrier, traffic cone, debris,
      bicycle rack, ...) -> ``UNKNOWN``.

    Anything unrecognised falls back to ``UNKNOWN`` with a warning, so a label a
    future release adds never crashes a conversion.
    """
    name = str(category_name)
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return DetectionType.BICYCLE
    if name.startswith("vehicle."):
        return DetectionType.VEHICLE
    if name.startswith("human.pedestrian") or name == "animal":
        return DetectionType.PEDESTRIAN
    if name.startswith("movable_object.") or name.startswith("static_object."):
        return DetectionType.UNKNOWN
    logging.warning("nuScenes: unmapped category %r -> UNKNOWN", name)
    return DetectionType.UNKNOWN


def box_to_ego_xyzhwl(
    box: io.NuscBox, t_ego_from_global: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Transform one global-frame box into the ego frame.

    Returns ``(x, y, z, heading, length, width, height)`` in the ego frame.
    nuScenes' ``size`` is ``(width, length, height)``; it is reordered here to
    StandardE2E's length/width/height. The yaw is read off the composed
    ego<-global<-box rotation, so road-grade pitch/roll in the ego pose is
    projected to a ground-plane heading consistently with the ego pose itself.
    """
    rotation = quat_wxyz_to_rotmat(box.rotation_wxyz)
    t_global_box = se3(rotation, box.center_global, dtype=np.float64)
    t_ego_box = t_ego_from_global @ t_global_box
    x, y, z, heading = matrix_to_xyz_heading(t_ego_box)
    width, length, height = (float(v) for v in box.wlh)
    return x, y, z, heading, length, width, height


class NuscenesDatasetProcessor(SourceDatasetProcessor):
    """Processor for nuScenes (v1.0).

    Frames arrive already resolved (see :class:`._nuscenes_io.NuscFrame`): the
    converter loads the JSON tables once in the parent and hands each worker a
    self-contained frame, so the processor holds no table state and only reads
    the per-frame pixels / points and transforms boxes into the ego frame.
    """

    DATASET_NAME = "nuscenes"

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
        # Per-worker cache of parsed vector maps, keyed by expansion-json path
        # (at most the 4 nuScenes locations); ``_warned_no_map`` makes the
        # "map-expansion not downloaded" warning fire once, not per frame.
        self._map_cache: dict[str, NuscMap] = {}
        self._warned_no_map = False

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # The official nuScenes split labels; each also fixes which metadata
        # version the converter loads (see ``_nuscenes_splits.VERSION_FOR_SPLIT``).
        return ["train", "val", "test", "mini_train", "mini_val"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            CamerasIdentityAdapter(),
            LidarAdapter(),
            Detections3DIdentityAdapter(),
            HDMapBEVAdapter(),
        ]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- modality builders -------------------------------------------------

    def _build_cameras(self, frame: io.NuscFrame) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for cam in frame.cameras:
            direction = _CHANNEL_TO_DIRECTION.get(cam.channel)
            if direction is None:
                logging.warning(
                    "nuScenes: camera channel %r is not a mapped CameraDirection;"
                    " skipping it.",
                    cam.channel,
                )
                continue
            if not os.path.exists(cam.image_path):
                logging.warning(
                    "nuScenes: camera file missing (partial download?): %s",
                    cam.image_path,
                )
                continue
            cameras[direction] = CameraData(
                camera_direction=direction,
                image=io.read_image(cam.image_path),
                intrinsics=cam.intrinsics,
                extrinsics=cam.extrinsics,
                is_fisheye=False,
            )
        return cameras

    def _build_lidar(self, frame: io.NuscFrame) -> Optional[LidarData]:
        if frame.lidar_path is None or frame.lidar_extrinsics is None:
            return None
        if not os.path.exists(frame.lidar_path):
            logging.warning(
                "nuScenes: lidar file missing (partial download?): %s",
                frame.lidar_path,
            )
            return None
        xyz_lidar = io.read_lidar_xyz(frame.lidar_path)
        xyz_ego = transform_points(frame.lidar_extrinsics, xyz_lidar)
        return LidarData(
            points=pd.DataFrame(xyz_ego, columns=[c.value for c in LidarComponent])
        )

    def _build_detections(
        self,
        frame: io.NuscFrame,
        t_ego_from_global: np.ndarray,
        timestamp_s: float,
    ) -> list[Detection3D]:
        detections: list[Detection3D] = []
        for box in frame.detections:
            x, y, z, heading, length, width, height = box_to_ego_xyzhwl(
                box, t_ego_from_global
            )
            detections.append(
                Detection3D(
                    unique_agent_id=box.instance_token,
                    detection_type=detection_type_for_category(box.category_name),
                    trajectory=Trajectory(
                        {
                            TC.TIMESTAMP: [timestamp_s],
                            TC.X: [x],
                            TC.Y: [y],
                            TC.Z: [z],
                            TC.HEADING: [heading],
                            TC.LENGTH: [length],
                            TC.WIDTH: [width],
                            TC.HEIGHT: [height],
                        }
                    ),
                )
            )
        return detections

    def _build_hd_map(self, frame: io.NuscFrame) -> Optional[HDMap]:
        """Vector map within the ego ROI, in the ego frame (``None`` if the
        map-expansion pack isn't present for this location)."""
        path = frame.map_expansion_path
        if path is None:
            if not self._warned_no_map:
                logging.warning(
                    "nuScenes: no map-expansion json for location %r; HD map "
                    "skipped. Unzip nuScenes-map-expansion-v1.3 into "
                    "<dataroot>/maps/ to enable it.",
                    frame.location,
                )
                self._warned_no_map = True
            return None
        nusc_map = self._map_cache.get(path)
        if nusc_map is None:
            nusc_map = NuscMap.from_json(path)
            self._map_cache[path] = nusc_map
        return nusc_map.build_hd_map(frame.pose_global_from_ego)

    # --- main entry point --------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        frame: io.NuscFrame = raw_frame_data
        pose = frame.pose_global_from_ego
        x, y, z, heading = matrix_to_xyz_heading(pose)
        timestamp_s = frame.timestamp_s

        cameras = (
            self._build_cameras(frame)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(frame)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )
        detections: list[Detection3D] = []
        if self.needs_attr(StandardFrameDataField.FRAME_DETECTIONS_3D):
            t_ego_from_global = np.linalg.inv(pose)
            detections = self._build_detections(frame, t_ego_from_global, timestamp_s)
        hd_map = (
            self._build_hd_map(frame)
            if self.needs_attr(StandardFrameDataField.HD_MAP)
            else None
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=frame.scene_name,
            frame_id=frame.frame_id,
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
            frame_detections_3d=FrameDetections3D(detections=detections),
            aux_data={"pose_matrix": pose.astype(np.float64)},
            extra_index_data={
                "scene": frame.scene_name,
                "sample_token": frame.sample_token,
            },
        )
