"""Source-dataset processor for View-of-Delft (VoD).

VoD (TU Delft, RA-L 2022) is a compact urban dataset whose distinctive sensor is
a **3+1D radar**; it also carries a 64-layer Velodyne LiDAR, a single front
camera and densely annotated KITTI-format 3D boxes over 24 short recording
scenes. The release ships as zips and must be extracted first (see
``scripts/extract_vod.sh`` / :mod:`._vod_io`).

Mapping to ``StandardFrameData`` (this release covers the camera, the Velodyne
LiDAR, the 3D detections and the ego trajectory):

* One **frame = one keyframe** (a global frame index); one **segment = one
  recording scene** (``delft_*``), grouped via the vendored official scene table
  (:mod:`._vod_splits`) so per-segment trajectories never span two recordings.
* **cameras**: the single front camera under :class:`CameraDirection.FRONT`.
  Intrinsics are the calib's ``P2`` 3x3 ``K``; extrinsics are
  ``T_velodyne_from_camera`` (``inv(Tr_velo_to_cam)``).
* **lidar**: the Velodyne ``.bin`` xyz, already in the velodyne (ego) frame
  (reflectance is dropped -- StandardE2E lidar is xyz-only).
* **frame_detections_3d**: the KITTI boxes mapped from camera coordinates into
  the ego frame, each ``category`` folded into the coarse ``DetectionType``
  taxonomy; ``DontCare`` is excluded. Box yaw is VoD's rotation about LiDAR -Z,
  so the ego-frame heading is its negation (see :mod:`._vod_geometry`).
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose
  ``T_map_from_lidar`` in the static map frame, from the per-frame pose JSON.
  Past/future ego trajectories are produced by
  :class:`FuturePastStatesFromMatricesAggregator`, as for AV2 / Waymo / TruckDrive.

Ego frame: ``velodyne`` (FLU -- x-forward, y-left, z-up), the frame the ``.bin``
points already live in and the frame VoD defines box yaw against, so points and
boxes are mutually consistent.

Not ingested (no StandardE2E target yet): the **3+1D radar** (the ``radar`` /
``radar_3frames`` / ``radar_5frames`` trees) and per-point LiDAR reflectance.
Timestamps are **synthesised** at the 10 Hz LiDAR lead rate from the frame index
(the detection release ships no per-frame timestamp); they order frames within a
segment and drive trajectory resampling, but are not the original capture times.
The **test** split ships sensor data without labels, so its frames carry no
detections.
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
from standard_e2e.caching.src_datasets.vod import _vod_io as io
from standard_e2e.caching.src_datasets.vod._vod_geometry import (
    box_camera_to_ego,
    ego_pose_map_from_lidar,
    parse_calibration,
)
from standard_e2e.caching.src_datasets.vod._vod_splits import ALLOWED_SPLITS
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
from standard_e2e.utils import matrix_to_xyz_heading

# Synthesised frame period: VoD syncs every sensor to the ~10 Hz LiDAR scan, and
# the detection release ships no per-frame timestamp, so frame index * this is
# used as the (segment-relative) timestamp.
_FRAME_PERIOD_S = 0.1

# VoD's 13 annotated classes (+ KITTI ``DontCare``) -> StandardE2E's coarse
# ``DetectionType``. ``None`` excludes the box. The two-wheeler family (the
# bicycle, its rider, mopeds/scooters and motorcycles) all fold into ``BICYCLE``
# -- the taxonomy's VRU-on-wheels bucket -- matching how VoD's own ``Cyclist``
# combines bike+rider. Static / non-agent boxes (a parked bicycle rack, a statue)
# and the genuinely-ambiguous "other"/"uncertain" rides fall to ``UNKNOWN``. VoD
# annotates no traffic signs, so ``SIGN`` is unused.
_CLASS_TO_DETECTION_TYPE: dict[str, Optional[DetectionType]] = {
    "Car": DetectionType.VEHICLE,
    "truck": DetectionType.VEHICLE,
    "vehicle_other": DetectionType.VEHICLE,
    "Pedestrian": DetectionType.PEDESTRIAN,
    "Cyclist": DetectionType.BICYCLE,
    "bicycle": DetectionType.BICYCLE,
    "rider": DetectionType.BICYCLE,
    "moped_scooter": DetectionType.BICYCLE,
    "motor": DetectionType.BICYCLE,
    "bicycle_rack": DetectionType.UNKNOWN,
    "human_depiction": DetectionType.UNKNOWN,
    "ride_other": DetectionType.UNKNOWN,
    "ride_uncertain": DetectionType.UNKNOWN,
    "DontCare": None,
}


def detection_type_for_class(class_name: str) -> Optional[DetectionType]:
    """Resolve a raw VoD ``class`` string to a ``DetectionType`` (``None`` ->
    exclude).

    Falls back to case-insensitive substring rules for labels absent from the
    vendored map so a future/unseen class never crashes a conversion:
    ``DontCare`` -> excluded; car/truck/van/bus/vehicle -> VEHICLE; the
    two-wheeler words -> BICYCLE; pedestrian -> PEDESTRIAN; otherwise UNKNOWN.
    """
    mapped = _CLASS_TO_DETECTION_TYPE.get(class_name, "__missing__")
    if mapped != "__missing__":
        return mapped  # type: ignore[return-value]
    name = str(class_name)
    if name == "DontCare":
        return None
    low = name.lower()
    if any(token in low for token in ("car", "truck", "van", "bus", "vehicle")):
        return DetectionType.VEHICLE
    if any(
        token in low
        for token in ("cyclist", "bicycle", "moped", "scooter", "motor", "rider")
    ):
        return DetectionType.BICYCLE
    if "pedestrian" in low:
        return DetectionType.PEDESTRIAN
    logging.warning("VoD: unmapped detection class %r -> UNKNOWN", name)
    return DetectionType.UNKNOWN


class VodDatasetProcessor(SourceDatasetProcessor):
    """Processor for the View-of-Delft dataset.

    Per-frame calibration and pose are tiny KITTI text/JSON files, so each frame
    is read independently -- there is no per-scene state to cache.
    """

    DATASET_NAME = "vod"

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
            Detections3DIdentityAdapter(),
        ]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- modality builders -------------------------------------------------

    def _build_cameras(
        self,
        ref: io.FrameRef,
        intrinsics: np.ndarray,
        t_lidar_from_cam: np.ndarray,
    ) -> dict[CameraDirection, CameraData]:
        return {
            CameraDirection.FRONT: CameraData(
                camera_direction=CameraDirection.FRONT,
                image=io.read_image(ref.image_path),
                intrinsics=intrinsics,
                extrinsics=t_lidar_from_cam.astype(np.float32),
                is_fisheye=False,
            )
        }

    def _build_lidar(self, ref: io.FrameRef) -> Optional[LidarData]:
        xyz_ego = io.read_velodyne_xyz(ref.velodyne_path)
        return LidarData(
            points=pd.DataFrame(xyz_ego, columns=[c.value for c in LidarComponent])
        )

    def _build_detections(
        self,
        ref: io.FrameRef,
        timestamp_s: float,
        t_lidar_from_cam: np.ndarray,
    ) -> list[Detection3D]:
        detections: list[Detection3D] = []
        for line in io.read_label_lines(ref.label_path):
            label = io.parse_label_line(line)
            if label is None:
                continue
            detection_type = detection_type_for_class(label.cls)
            if detection_type is None:
                continue  # DontCare
            center, length, width, height, heading = box_camera_to_ego(
                label.location_cam,
                label.dimensions_hwl,
                label.rotation,
                t_lidar_from_cam,
            )
            detections.append(
                Detection3D(
                    # Stable across frames only when the label_2_with_track_ids
                    # set is installed; otherwise this is KITTI's (unused)
                    # truncation field.
                    unique_agent_id=label.track_field,
                    detection_type=detection_type,
                    trajectory=Trajectory(
                        {
                            TC.TIMESTAMP: [timestamp_s],
                            TC.X: [float(center[0])],
                            TC.Y: [float(center[1])],
                            TC.Z: [float(center[2])],
                            TC.HEADING: [heading],
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
        ref: io.FrameRef = raw_frame_data
        intrinsics, t_cam_from_lidar = parse_calibration(
            io.read_calibration_text(ref.calib_path)
        )
        t_lidar_from_cam = np.linalg.inv(t_cam_from_lidar)

        pose = io.read_pose(ref.pose_path)
        map_from_camera = pose.get("mapToCamera")
        if map_from_camera is None:
            raise KeyError(f"{ref.pose_path}: missing 'mapToCamera' transform")
        pose_map_from_lidar = ego_pose_map_from_lidar(map_from_camera, t_cam_from_lidar)
        x, y, z, heading = matrix_to_xyz_heading(pose_map_from_lidar)
        timestamp_s = ref.frame_id * _FRAME_PERIOD_S

        cameras = (
            self._build_cameras(ref, intrinsics, t_lidar_from_cam)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(ref)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )
        detections = (
            self._build_detections(ref, timestamp_s, t_lidar_from_cam)
            if self.needs_attr(StandardFrameDataField.FRAME_DETECTIONS_3D)
            else []
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.scene_name,
            frame_id=ref.frame_id,
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
            aux_data={"pose_matrix": pose_map_from_lidar.astype(np.float64)},
            extra_index_data={"scene": ref.scene_name, "frame_index": ref.frame_id},
        )
