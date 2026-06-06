"""Source-dataset processor for TruckDrive.

TruckDrive (Torc Robotics / Princeton, CVPR 2026) is a long-range highway truck
dataset: per scene a ~15-25 s synchronized stream from an 11-camera 8 MP
surround rig, seven long-range Aeva FMCW LiDARs (delivered as one joint cloud),
short-range Ousters, 4D radars, PPK+SLAM ego poses and densely tracked 3D
boxes. Scenes ship as per-modality zips and must be extracted first (see
``_truckdrive_io`` / ``scripts/extract_truckdrive.sh``).

Mapping to ``StandardFrameData`` (this first release covers cameras, the ego
trajectory, the Aeva LiDAR and 3D detections; radar, accumulated-depth and lane
lines have no StandardE2E target yet):

* One **frame = one synchronization key** (``sync_id``) that carries an ego
  pose; one **scene = one cache segment**. Cameras / LiDAR / boxes are attached
  by ``sync_id`` when present (a sensor that did not fire for a given snapshot
  is simply absent that frame).
* **cameras**: every physical camera present in the scene, keyed by
  :class:`CameraDirection` -- the canonical surround member matching its facing
  (so keys mean the same thing across datasets), with dedicated members only for
  the extra forward-telephoto and rear-facing side cameras the eight canonical
  slots cannot name. Intrinsics are the rectified pinhole ``K``; extrinsics are
  ``T_velodyne_from_camera`` from the static transform tree.
* **lidar**: the Aeva joint cloud's xyz, transformed from the Aeva reference
  frame into the ego (``velodyne``) frame.
* **frame_detections_3d**: the tracked 3D boxes in the ego frame. The ego
  vehicle's own cab/trailer boxes (``Vehicle-EgoVehicle-*``) and ``DontCare``
  groups are excluded, matching the paper's evaluation taxonomy.
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose
  ``T_world_from_ego`` in the per-scene local world frame. Past/future ego
  trajectories are produced by :class:`FuturePastStatesFromMatricesAggregator`,
  exactly as for AV2 / Waymo / WayveScenes.

Ego frame: ``velodyne`` -- the frame the public 3D boxes and the devkit viewer
use (FLU). LiDAR is moved into it via the calibration tree and detections are
read from the annotations' ego-frame fields, so points and boxes are mutually
consistent. The ego pose is treated as ``T_world_from_velodyne``; any residual
cab/sensor lever-arm is absorbed by using ``velodyne`` consistently throughout.
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
    Detections3DIdentityAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.truckdrive import _truckdrive_io as io
from standard_e2e.caching.src_datasets.truckdrive._truckdrive_geometry import (
    AEVA_REFERENCE_FRAME,
    VELODYNE_FRAME,
    build_tf_graph,
    find_transform,
    pose_world_from_ego,
)
from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
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
from standard_e2e.utils import matrix_to_xyz_heading, transform_points

# TruckDrive's 9 evaluation categories -> StandardE2E's coarse DetectionType.
# ``None`` marks categories dropped from detections (the ego vehicle and the
# DontCare groups), per supplementary Sec. 3 / Table 2.
_CATEGORY_TO_DETECTION_TYPE: dict[str, Optional[DetectionType]] = {
    "DontCare": None,
    "Two-Wheel": DetectionType.BICYCLE,
    "Passenger-Car": DetectionType.VEHICLE,
    "Person": DetectionType.PEDESTRIAN,
    "RoadObstruction": DetectionType.UNKNOWN,
    "SemiTruck-Cab": DetectionType.VEHICLE,
    "SemiTruck-Trailer": DetectionType.VEHICLE,
    "Vehicle": DetectionType.VEHICLE,
    "TrafficSign": DetectionType.SIGN,
    "EmergencyVehicle": DetectionType.VEHICLE,
}

# Fine-grained ``class-id`` -> evaluation category (supplementary Table 2). The
# flattened ``class-id -> DetectionType`` map is derived from this at import.
_CATEGORY_TO_CLASS_IDS: dict[str, list[str]] = {
    "DontCare": [
        "DelineatorGroupDontCare",
        "OutOfLidarRangeVehicleGroup",
        "ParkingLotVehicleGroup",
        "Vehicle-EgoVehicle-Cab",
        "Vehicle-EgoVehicle-Trailer",
    ],
    "Two-Wheel": [
        "VRUvehicle-Bicycle",
        "VRUvehicle-Motorcycle",
        "VRUvehicle-StandingScooter",
        "VRUvehicle-Wheelchair",
        "Vehicle-Bicycle",
        "Vehicle-Motorcycle",
    ],
    "Passenger-Car": ["Vehicle-Passenger"],
    "Person": [
        "Animal",
        "Person",
        "Person-Other",
        "Person-Pedestrian",
        "Person-Rider",
        "Person-Skater",
        "Person-TrafficControl",
    ],
    "RoadObstruction": [
        "RoadDebris",
        "RoadDebris-Other",
        "RoadDebris-Pothole",
        "RoadDebris-RoadKill",
        "RoadDebris-Tire",
        "RoadDebris-Vegetation",
        "RoadObstruction",
        "RoadObstruction-Barrel",
        "RoadObstruction-Barricade",
        "RoadObstruction-Cone",
        "RoadObstruction-Delineator",
        "RoadObstruction-Other",
        "RoadObstruction-VerticalPanel",
    ],
    "SemiTruck-Cab": ["Vehicle-SemiTruck-Cab"],
    "SemiTruck-Trailer": ["Vehicle-SemiTruck-Trailer"],
    "Vehicle": [
        "VRUvehicle",
        "VRUvehicle-Other",
        "VRUvehicle-Trailer",
        "Vehicle",
        "Vehicle-Bus",
        "Vehicle-DeliveryVan",
        "Vehicle-Equipment",
        "Vehicle-HeavyVehicle",
        "Vehicle-Other",
        "Vehicle-RV",
        "Vehicle-SchoolBus",
        "Vehicle-SingleUnitTruck",
        "Vehicle-Trailer",
        "Vehicle-Unibody",
    ],
    "EmergencyVehicle": ["Vehicle-Emergency", "Vehicle-Police"],
}

# Flattened ``class-id -> DetectionType | None`` (None -> excluded).
_CLASS_ID_TO_DETECTION_TYPE: dict[str, Optional[DetectionType]] = {
    class_id: _CATEGORY_TO_DETECTION_TYPE[category]
    for category, class_ids in _CATEGORY_TO_CLASS_IDS.items()
    for class_id in class_ids
}


def detection_type_for_class_id(class_id: str) -> Optional[DetectionType]:
    """Resolve a raw ``class-id`` to a ``DetectionType`` (``None`` -> exclude).

    Falls back to prefix rules for fine labels not in the vendored Table-2 map
    (TrafficSign* / TrafficSignal* / *Signal -> SIGN; bare ``Vehicle*`` ->
    VEHICLE; ``EgoVehicle`` / ``DontCare`` -> excluded; otherwise UNKNOWN) so a
    label that the public scenes don't contain never crashes a conversion.
    """
    mapped = _CLASS_ID_TO_DETECTION_TYPE.get(class_id, "__missing__")
    if mapped != "__missing__":
        return mapped  # type: ignore[return-value]
    name = str(class_id)
    if "EgoVehicle" in name or name.startswith("DontCare"):
        return None
    low = name.lower()
    if "trafficsign" in low or "trafficsignal" in low or "signal" in low:
        return DetectionType.SIGN
    if "bicycle" in low or "motorcycle" in low or "scooter" in low:
        return DetectionType.BICYCLE
    if name.startswith("Person") or name == "Animal":
        return DetectionType.PEDESTRIAN
    if name.startswith("RoadObstruction") or name.startswith("RoadDebris"):
        return DetectionType.UNKNOWN
    if name.startswith("Vehicle") or name.startswith("VRUvehicle"):
        return DetectionType.VEHICLE
    logging.warning("TruckDrive: unmapped detection class-id %r -> UNKNOWN", name)
    return DetectionType.UNKNOWN


# Physical rig camera (``camera/leopard/<name>``) -> canonical CameraDirection.
# The canonical surround members are reused wherever a camera matches that
# facing; only the genuinely-extra views (the forward telephoto pair and the
# rear-facing side pair) get dedicated members. The full 15-camera rig adds
# ``forward_{left,right}_medium`` and ``rearward_{left,right}_top_medium``,
# which are absent from the public scenes and so are left unmapped (skipped).
_CAMERA_NAME_TO_DIRECTION: dict[str, CameraDirection] = {
    "forward_center_medium": CameraDirection.FRONT,
    "forward_left_wide": CameraDirection.FRONT_LEFT,
    "forward_right_wide": CameraDirection.FRONT_RIGHT,
    "forward_left_narrow": CameraDirection.FRONT_LEFT_NARROW,
    "forward_right_narrow": CameraDirection.FRONT_RIGHT_NARROW,
    "sideward_left_front_wide": CameraDirection.SIDE_LEFT,
    "sideward_right_front_wide": CameraDirection.SIDE_RIGHT,
    "sideward_left_back_wide": CameraDirection.SIDE_LEFT_BACK,
    "sideward_right_back_wide": CameraDirection.SIDE_RIGHT_BACK,
    "rearward_left_bottom_medium": CameraDirection.REAR_LEFT,
    "rearward_right_bottom_medium": CameraDirection.REAR_RIGHT,
}


def _camera_direction_for(name: str) -> Optional[CameraDirection]:
    """Resolve a physical camera folder name to its ``CameraDirection``, or
    ``None`` if the rig camera is not mapped (e.g. a 15-rig extra)."""
    return _CAMERA_NAME_TO_DIRECTION.get(name)


def _box_value(obj: dict[str, Any], *keys: str) -> Optional[float]:
    """First finite float among ``obj[key]`` for ``keys`` in order, else None."""
    for key in keys:
        if key in obj:
            try:
                value = float(obj[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return None


class TruckDriveDatasetProcessor(SourceDatasetProcessor):
    """Processor for the TruckDrive dataset.

    Per-scene state (calibration, transform tree, ego-pose table and the
    per-sensor ``sync_id -> path`` maps) is read once and reused across every
    frame of the scene; the cache is keyed by ``scene_id`` so pooled workers
    that hop between scenes only reload on transition.
    """

    DATASET_NAME = "truckdrive"

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
        self._pose_by_sync: dict[int, tuple[float, np.ndarray]] = {}
        self._camera_intrinsics: dict[CameraDirection, np.ndarray] = {}
        self._camera_extrinsics: dict[CameraDirection, np.ndarray] = {}
        self._camera_paths: dict[CameraDirection, dict[int, str]] = {}
        self._aeva_paths: dict[int, str] = {}
        self._box_paths: dict[int, str] = {}
        self._velodyne_from_aeva: np.ndarray = np.eye(4, dtype=np.float32)

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # TruckDrive's public release scenes are all validation scenes; ``split``
        # is a passthrough output label here (the caller chooses it).
        return ["all", "train", "val", "test"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            CamerasIdentityAdapter(),
            LidarAdapter(),
            Detections3DIdentityAdapter(),
        ]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- per-scene cache ---------------------------------------------------

    def _refresh_scene_cache(self, ref: io.SceneRef) -> None:
        if self._cached_scene_id == ref.scene_id:
            return
        tf_tree, camera_calibs = io.read_calibrations(ref)
        graph = build_tf_graph(tf_tree)

        # Ego pose per sync_id (T_world_from_ego) + its timestamp.
        sync_ids, timestamps_s, poses_raw = io.read_trajectory(ref)
        self._pose_by_sync = {
            sync_id: (
                float(timestamps_s[i]),
                pose_world_from_ego(poses_raw[i, 0:3], poses_raw[i, 3:7]),
            )
            for i, sync_id in enumerate(sync_ids)
        }

        # Cameras present in this scene -> intrinsics, extrinsics, file maps.
        self._camera_intrinsics = {}
        self._camera_extrinsics = {}
        self._camera_paths = {}
        for name in io.camera_names(ref):
            direction = _camera_direction_for(name)
            if direction is None:
                logging.warning(
                    "TruckDrive: camera %r is not an enumerated CameraDirection;"
                    " skipping it.",
                    name,
                )
                continue
            if name not in camera_calibs:
                logging.warning("TruckDrive: no calibration for camera %r", name)
                continue
            self._camera_intrinsics[direction] = camera_calibs[name]["intrinsics"]
            self._camera_extrinsics[direction] = find_transform(
                graph, f"camera_leopard_{name}", VELODYNE_FRAME
            ).astype(np.float32)
            self._camera_paths[direction] = io.camera_sync_paths(ref, name)

        self._aeva_paths = io.aeva_sync_paths(ref)
        self._box_paths = io.box_sync_paths(ref)
        self._velodyne_from_aeva = find_transform(
            graph, AEVA_REFERENCE_FRAME, VELODYNE_FRAME
        ).astype(np.float32)

        self._cached_scene_id = ref.scene_id
        logging.debug(
            "TruckDrive scene %s cached: %d poses, %d cameras",
            ref.scene_id,
            len(self._pose_by_sync),
            len(self._camera_paths),
        )

    # --- modality builders -------------------------------------------------

    def _build_cameras(self, sync_id: int) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for direction, sync_paths in self._camera_paths.items():
            path = sync_paths.get(sync_id)
            if path is None:
                continue
            cameras[direction] = CameraData(
                camera_direction=direction,
                image=io.read_image(path),
                intrinsics=self._camera_intrinsics[direction],
                extrinsics=self._camera_extrinsics[direction],
                is_fisheye=False,
            )
        return cameras

    def _build_lidar(self, sync_id: int) -> Optional[LidarData]:
        path = self._aeva_paths.get(sync_id)
        if path is None:
            return None
        xyz_aeva = io.read_aeva_xyz(path)
        xyz_ego = transform_points(self._velodyne_from_aeva, xyz_aeva)
        return LidarData(
            points=pd.DataFrame(xyz_ego, columns=[c.value for c in LidarComponent])
        )

    def _build_detections(self, sync_id: int, timestamp_s: float) -> list[Detection3D]:
        path = self._box_paths.get(sync_id)
        if path is None:
            return []
        detections: list[Detection3D] = []
        for obj in io.read_boxes(path):
            detection_type = detection_type_for_class_id(obj.get("class-id", ""))
            if detection_type is None:
                continue  # ego vehicle / DontCare
            x = _box_value(obj, "x", "x_ego")
            y = _box_value(obj, "y", "y_ego")
            z = _box_value(obj, "z", "z_ego")
            length = _box_value(obj, "l", "l_ego", "length")
            width = _box_value(obj, "w", "w_ego", "width")
            height = _box_value(obj, "h", "h_ego")
            yaw = _box_value(obj, "yaw", "yaw_ego", "yaw_rad")
            # Skip a box missing any geometry field (also narrows the Optionals
            # below to float for the type checker).
            if (
                x is None
                or y is None
                or z is None
                or length is None
                or width is None
                or height is None
                or yaw is None
            ):
                continue
            detections.append(
                Detection3D(
                    unique_agent_id=str(
                        obj.get("Tracking_ID", obj.get("id", "unknown"))
                    ),
                    detection_type=detection_type,
                    trajectory=Trajectory(
                        {
                            TC.TIMESTAMP: [timestamp_s],
                            TC.X: [x],
                            TC.Y: [y],
                            TC.Z: [z],
                            TC.HEADING: [yaw],
                            TC.LENGTH: [abs(length)],
                            TC.WIDTH: [abs(width)],
                            TC.HEIGHT: [abs(height)],
                        }
                    ),
                )
            )
        return detections

    # --- main entry point --------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        ref, sync_id = raw_frame_data
        self._refresh_scene_cache(ref)
        pose_entry = self._pose_by_sync.get(sync_id)
        if pose_entry is None:
            raise KeyError(f"sync_id {sync_id} has no ego pose in scene {ref.scene_id}")
        timestamp_s, pose_world_from_velodyne = pose_entry
        x, y, z, heading = matrix_to_xyz_heading(pose_world_from_velodyne)

        cameras = (
            self._build_cameras(sync_id)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(sync_id)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )
        detections = (
            self._build_detections(sync_id, timestamp_s)
            if self.needs_attr(StandardFrameDataField.FRAME_DETECTIONS_3D)
            else []
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.scene_id,
            frame_id=sync_id,
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
            frame_detections_3d=FrameDetections3D(detections=detections),
            aux_data={"pose_matrix": pose_world_from_velodyne.astype(np.float64)},
            extra_index_data={"scene": ref.scene_id, "sync_id": sync_id},
        )
