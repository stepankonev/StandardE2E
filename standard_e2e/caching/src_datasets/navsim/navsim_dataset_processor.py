"""Source-dataset processor for NAVSIM (OpenScene-v1.1).

NAVSIM ships scenes as per-log pickle files (lists of frame dicts) plus
a parallel ``sensor_blobs`` tree with the actual camera JPEGs and merged
lidar PCDs. We read the dicts directly and produce
:class:`StandardFrameData`, sidestepping the heavy ``nuplan-devkit``
chain — none of NAVSIM's API surface is required for the data path we
need.

**Phase 1 modality coverage** (this commit): cameras (8), lidar
(merged sweep), 3D detections, driving command (Intent), past/future
ego trajectory via segment-context aggregator. **HD map is deferred**
to a follow-up: it requires loading nuPlan ``map.gpkg`` files via
``nuplan.common.maps`` and translating them through the unified
:class:`MapElementType` taxonomy, which is a non-trivial design
exercise on its own.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    IntentIdentityAdapter,
    LidarAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.navsim._pcd import read_navsim_pcd_xyz
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
    Intent,
    LidarComponent,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import matrix_to_xyz_heading

# NAVSIM ships 8 cameras: front-centre, three on each side (front-of-side,
# side, rear-of-side), plus rear-centre. Mapping into our 8-direction
# CameraDirection enum is exact.
_NAVSIM_CAM_TO_DIRECTION: dict[str, CameraDirection] = {
    "CAM_F0": CameraDirection.FRONT,
    "CAM_L0": CameraDirection.FRONT_LEFT,
    "CAM_R0": CameraDirection.FRONT_RIGHT,
    "CAM_L1": CameraDirection.SIDE_LEFT,
    "CAM_R1": CameraDirection.SIDE_RIGHT,
    "CAM_L2": CameraDirection.REAR_LEFT,
    "CAM_R2": CameraDirection.REAR_RIGHT,
    "CAM_B0": CameraDirection.REAR,
}

# NAVSIM detection labels. The dataset uses lowercase short strings
# (vehicle/pedestrian/bicycle/...). Anything we don't recognise lands in
# DetectionType.UNKNOWN so the framework still surfaces it.
_NAVSIM_CATEGORY_TO_DETECTION_TYPE: dict[str, DetectionType] = {
    "vehicle": DetectionType.VEHICLE,
    "pedestrian": DetectionType.PEDESTRIAN,
    "bicycle": DetectionType.BICYCLE,
    "traffic_cone": DetectionType.UNKNOWN,
    "barrier": DetectionType.UNKNOWN,
    "czone_sign": DetectionType.SIGN,
    "generic_object": DetectionType.UNKNOWN,
}

# NAVSIM driving_command is a (4,) one-hot. Index → Intent mapping is
# documented at navsim/docs/agents.md: "left, straight, right, unknown".
# Real-data distribution (~7%/90%/2%/0%) confirms this layout.
_DRIVING_COMMAND_TO_INTENT: tuple[Intent, ...] = (
    Intent.GO_LEFT,
    Intent.GO_STRAIGHT,
    Intent.GO_RIGHT,
    Intent.UNKNOWN,
)


def _quat_to_rotmat_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """NAVSIM (and nuplan) store Hamilton quaternions as (qw, qx, qy, qz)."""
    rotmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    return cast(np.ndarray, rotmat)


def _se3_from_rotation_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(rotation, dtype=np.float32)
    T[:3, 3] = np.asarray(translation, dtype=np.float32)
    return T


def _driving_command_to_intent(driving_command: np.ndarray) -> Intent:
    """Decode the 4-element one-hot into our Intent enum.

    Off-spec inputs (multiple ones, all zeros, NaN) fall back to UNKNOWN
    rather than asserting — preprocessing should keep going through
    rare malformed frames.
    """
    arr = np.asarray(driving_command).astype(np.int64).flatten()
    if arr.shape != (4,) or arr.sum() != 1:
        return Intent.UNKNOWN
    idx = int(np.argmax(arr))
    return _DRIVING_COMMAND_TO_INTENT[idx]


class NavsimDatasetProcessor(SourceDatasetProcessor):
    """Processor for NAVSIM (OpenScene-v1.1 trainval).

    A "frame" is one ego timestamp inside a scene pickle. Per-log state
    (the entire pickle) is loaded once and reused for every frame in
    that log; the cache key is the pickle path, so workers that hop
    between logs only reload at log boundaries.

    See ``Phase 1`` notes in the module docstring — HD map is not yet
    surfaced.
    """

    DATASET_NAME = "navsim"

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
        self._cached_log_path: Optional[Path] = None
        self._frames: list[dict[str, Any]] = []

    @property
    def allowed_splits(self) -> list[str]:
        # OpenScene-v1.1 ships a single 'trainval' split; NAVSIM users sample
        # train/val/test subsets from it via SceneFilter.log_names. We expose
        # 'trainval' verbatim so the converter can find the corresponding
        # navsim_logs/<split>/ and sensor_blobs/<split>/ trees.
        return ["trainval"]

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            CamerasIdentityAdapter(),
            LidarAdapter(),
            IntentIdentityAdapter(),
            Detections3DIdentityAdapter(),
        ]

    def _get_default_context_aggregators(self):
        return [
            FuturePastStatesFromMatricesAggregator(self.output_path),
            FutureDetectionsAggregator(self.output_path),
        ]

    # --- per-log cache -------------------------------------------------------

    def _refresh_log_cache(self, log_path: Path) -> None:
        if self._cached_log_path == log_path:
            return
        logging.info("Loading NAVSIM scene pickle %s", log_path.name)
        with open(log_path, "rb") as fp:
            self._frames = pickle.load(fp)
        self._cached_log_path = log_path

    # --- per-frame helpers ---------------------------------------------------

    def _build_camera_dict(
        self, sensor_blobs_root: Path, frame: dict[str, Any]
    ) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for cam_name, direction in _NAVSIM_CAM_TO_DIRECTION.items():
            cam_dict = frame["cams"].get(cam_name)
            if cam_dict is None or "data_path" not in cam_dict:
                continue
            img_path = sensor_blobs_root / cam_dict["data_path"]
            if not img_path.exists():
                continue
            image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            extrinsics = _se3_from_rotation_translation(
                cam_dict["sensor2lidar_rotation"],
                cam_dict["sensor2lidar_translation"],
            )
            distortion = cam_dict.get("distortion")
            cameras[direction] = CameraData(
                image=image,
                camera_direction=direction,
                intrinsics=np.asarray(cam_dict["cam_intrinsic"], dtype=np.float32),
                extrinsics=extrinsics,
                distortion=(
                    np.asarray(distortion, dtype=np.float32)
                    if distortion is not None
                    else None
                ),
            )
        return cameras

    def _build_lidar(
        self, sensor_blobs_root: Path, frame: dict[str, Any]
    ) -> Optional[LidarData]:
        lidar_path = frame.get("lidar_path")
        if lidar_path is None:
            return None
        full_path = sensor_blobs_root / lidar_path
        if not full_path.exists():
            return None
        xyz = read_navsim_pcd_xyz(full_path)
        return LidarData(
            points=pd.DataFrame(xyz, columns=[c.value for c in LidarComponent])
        )

    def _build_detections(
        self, frame: dict[str, Any], timestamp_s: float
    ) -> list[Detection3D]:
        anns = frame.get("anns")
        if anns is None:
            return []
        boxes = np.asarray(anns.get("gt_boxes"), dtype=np.float64)
        names = list(anns.get("gt_names", []))
        track_tokens = list(anns.get("track_tokens", []))
        if len(boxes) == 0:
            return []
        if not (len(boxes) == len(names) == len(track_tokens)):
            raise ValueError(
                f"Inconsistent annotation lengths in frame: "
                f"boxes={len(boxes)} names={len(names)} tracks={len(track_tokens)}"
            )
        detections: list[Detection3D] = []
        for box, name, track_token in zip(boxes, names, track_tokens):
            x, y, z, length, width, height, heading = (float(v) for v in box)
            detection_type = _NAVSIM_CATEGORY_TO_DETECTION_TYPE.get(
                str(name), DetectionType.UNKNOWN
            )
            detections.append(
                Detection3D(
                    unique_agent_id=str(track_token),
                    detection_type=detection_type,
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

    # --- main entry point ----------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        log_path, frame_idx = raw_frame_data
        self._refresh_log_cache(log_path)
        frame = self._frames[frame_idx]
        log_name = frame["log_name"]
        token = frame["token"]
        scene_token = frame["scene_token"]
        # NAVSIM ships per-frame microsecond timestamps. ``frame['frame_idx']``
        # is per-SCENE (cycles 0..N within each scene window) so it duplicates
        # across the ~11 scenes per pickle and is unsuitable as the segment
        # clock. ``frame['timestamp']`` is globally monotonic across the log.
        timestamp_s = float(frame["timestamp"]) * 1e-6

        # log_path is <input_path>/navsim_logs/<split>/<log>.pkl;
        # ascend three levels to <input_path>, then descend into sensor_blobs.
        sensor_blobs_root = log_path.parent.parent.parent / "sensor_blobs" / self._split

        cameras = self._build_camera_dict(sensor_blobs_root, frame)
        lidar = self._build_lidar(sensor_blobs_root, frame)
        detections = self._build_detections(frame, timestamp_s)
        intent = _driving_command_to_intent(frame["driving_command"])

        ego_translation = np.asarray(frame["ego2global_translation"], dtype=np.float32)
        ego_rotation = _quat_to_rotmat_wxyz(*frame["ego2global_rotation"])
        T_global_from_ego = _se3_from_rotation_translation(
            ego_rotation, ego_translation
        )
        x, y, z, heading = matrix_to_xyz_heading(T_global_from_ego)

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=str(log_name),
            # frame_idx cycles per-scene; the microsecond timestamp is the
            # only globally-unique ID within a log.
            frame_id=int(frame["timestamp"]),
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
            intent=intent,
            frame_detections_3d=FrameDetections3D(detections=detections),
            aux_data={"pose_matrix": T_global_from_ego},
            extra_index_data={
                "log_name": str(log_name),
                "scene_token": str(scene_token),
                "frame_token": str(token),
            },
        )
