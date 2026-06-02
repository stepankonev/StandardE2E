"""Source-dataset processor for WayveScenes101.

WayveScenes101 is a camera-centric driving dataset: 5 fisheye cameras
(forward + side arc), ~200 timestamps per scene, plus a per-scene COLMAP
``rig`` reconstruction (camera intrinsics + per-image world poses + a sparse
SfM point cloud). It ships **no** lidar, HD map, 3D boxes, or driving command.

Mapping to ``StandardFrameData``:

* One frame = one timestamp of a scene; one scene = one segment.
* **cameras**: the (up to) 5 images at that timestamp, with intrinsics and
  ego-relative extrinsics.
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose in a world
  FLU frame, derived from the front-forward camera's COLMAP pose. The
  past/future ego trajectories are produced by
  :class:`FuturePastStatesFromMatricesAggregator`, exactly as for AV2 / Waymo.
* **lidar**: the scene's sparse SfM point cloud, transformed into the
  per-frame ego frame and range-clipped — so it flows through the existing
  ``lidar_pc`` / ``lidar_bev`` adapters like a real sweep. It is *photogrammetric*,
  not a sensor measurement (see ``docs`` / commit message), but it is geometrically
  consistent with the lidar modality.

Coordinate frames: COLMAP is OpenCV/RDF; Wayve is FLU (x-forward, y-left,
z-up) — the same convention our lidar uses. The ``_FLU_RDF`` matrix and the
``_opencv_points_to_flu`` helper replicate the official SDK's conversions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    LidarAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.wayve_scenes._colmap import (
    qvec_wxyz_to_rotmat,
    read_cameras_bin,
    read_images_bin,
    read_points3D_bin,
)
from standard_e2e.data_structures import (
    CameraData,
    LidarData,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import (
    CameraDirection,
    LidarComponent,
    StandardFrameDataField,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import matrix_to_xyz_heading


def _decode_rgb(path: Path) -> np.ndarray:
    """JPEG -> contiguous uint8 RGB ndarray (cv2; no TF in the worker path)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


# WayveScenes camera-rig directory name -> our canonical CameraDirection.
_WAYVE_CAM_TO_DIRECTION: dict[str, CameraDirection] = {
    "front-forward": CameraDirection.FRONT,
    "left-forward": CameraDirection.FRONT_LEFT,
    "right-forward": CameraDirection.FRONT_RIGHT,
    "left-backward": CameraDirection.REAR_LEFT,
    "right-backward": CameraDirection.REAR_RIGHT,
}
# Reference camera that defines the ego frame for each timestamp.
_REFERENCE_CAMERA = "front-forward"

# 4x4 OpenCV-RDF -> Wayve-FLU world-axis reorientation (from the SDK's
# ``opencv_pose_to_wayve_pose``). Rows pick (z, -x, -y).
_FLU_RDF = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
_FLU_RDF_INV = np.linalg.inv(_FLU_RDF)


def _opencv_points_to_flu(points_opencv: np.ndarray) -> np.ndarray:
    """(N,3) OpenCV-world points -> Wayve-FLU world (matches SDK)."""
    flu = points_opencv[:, [2, 0, 1]].copy()
    flu[:, 1:] *= -1.0
    return flu


def _cam_from_world_4x4(qvec_wxyz: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = qvec_wxyz_to_rotmat(qvec_wxyz)
    T[:3, 3] = tvec
    return T


def _world_from_cam_flu(qvec_wxyz: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP cam_from_world -> FLU-body to FLU-world rigid transform.

    Conjugates the OpenCV camera->world pose by ``_FLU_RDF`` so both the world
    axes *and* the body axes are FLU (x-forward, y-left, z-up). The translation
    equals ``_opencv_points_to_flu(camera_center)``.
    """
    g_cam_world = _cam_from_world_4x4(qvec_wxyz, tvec)
    g_world_cam = np.linalg.inv(g_cam_world)
    return _FLU_RDF @ g_world_cam @ _FLU_RDF_INV


class WayveScenesDatasetProcessor(SourceDatasetProcessor):
    """Processor for the WayveScenes101 dataset."""

    DATASET_NAME = "wayve_scenes"
    # Left-to-right panorama order across the 5-camera arc.
    CAMERAS_ORDER = [
        CameraDirection.REAR_LEFT,
        CameraDirection.FRONT_LEFT,
        CameraDirection.FRONT,
        CameraDirection.FRONT_RIGHT,
        CameraDirection.REAR_RIGHT,
    ]
    # Range clip (metres) applied to the per-frame SfM cloud, to mimic a sweep.
    LIDAR_MAX_RANGE_M = 50.0

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
        # Per-scene cache (one scene = one segment), keyed by scene dir.
        self._cached_scene_dir: Optional[Path] = None
        # camera_id -> (K 3x3 float32, distortion (4,) float32)
        self._cam_calib: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # timestamp -> {camera_position -> (image_path, T_world_cam_flu 4x4)}
        self._frame_cams: dict[int, dict[str, tuple[Path, np.ndarray]]] = {}
        # timestamp -> ego pose T_world_ego (FLU, 4x4)
        self._ego_pose: dict[int, np.ndarray] = {}
        # Filtered scene SfM points in FLU world frame (N, 3) float32.
        self._points_world_flu = np.zeros((0, 3), dtype=np.float32)

    @property
    def allowed_splits(self) -> list[str]:
        # WayveScenes101 has no perception train/val/test split; the split is a
        # passthrough output label. Accept the common ones.
        return ["test", "train", "val", "all"]

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            PanoImageAdapter(cameras_order=self.CAMERAS_ORDER),
            LidarAdapter(),
        ]

    def _get_default_context_aggregators(self):
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- per-scene cache --------------------------------------------------

    def _refresh_scene_cache(self, scene_dir: Path) -> None:
        if self._cached_scene_dir == scene_dir:
            return
        rig = scene_dir / "colmap_sparse" / "rig"
        cameras = read_cameras_bin(rig / "cameras.bin")
        images = read_images_bin(rig / "images.bin")
        xyz, _rgb, error, track_len = read_points3D_bin(rig / "points3D.bin")

        # Camera intrinsics + fisheye distortion per camera_id.
        self._cam_calib = {}
        for cid, cam in cameras.items():
            fx, fy, cx, cy = cam.params[:4]
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            dist = cam.params[4:8].astype(np.float32)  # OPENCV_FISHEYE k1..k4
            self._cam_calib[cid] = (K, dist)

        # Group images by timestamp + camera position; precompute FLU poses.
        self._frame_cams = {}
        self._ego_pose = {}
        cam_pose_world: dict[int, dict[str, np.ndarray]] = {}
        for img in images:
            position = Path(img.name).parent.name
            ts = int(Path(img.name).stem)
            T_world_cam = _world_from_cam_flu(img.qvec_wxyz, img.tvec)
            self._frame_cams.setdefault(ts, {})[position] = (
                scene_dir / "images" / img.name,
                T_world_cam,
            )
            cam_pose_world.setdefault(ts, {})[position] = T_world_cam

        # Ego pose per timestamp from the reference camera.
        for ts, by_cam in cam_pose_world.items():
            if _REFERENCE_CAMERA in by_cam:
                self._ego_pose[ts] = by_cam[_REFERENCE_CAMERA]

        # SfM points: replicate the SDK filter (error <= 6, track length >= 2),
        # convert to FLU world once for the whole scene.
        keep = (error <= 6.0) & (track_len >= 2)
        self._points_world_flu = _opencv_points_to_flu(xyz[keep]).astype(np.float32)

        # Map camera_id per position (constant across the scene) for intrinsics.
        self._position_camera_id: dict[str, int] = {}
        for img in images:
            position = Path(img.name).parent.name
            self._position_camera_id.setdefault(position, img.camera_id)

        self._cached_scene_dir = scene_dir
        logging.info(
            "WayveScenes scene %s: %d timestamps, %d SfM points (FLU)",
            scene_dir.name,
            len(self._ego_pose),
            len(self._points_world_flu),
        )

    # --- modality builders ------------------------------------------------

    def _build_cameras(
        self, ts: int, T_ego_world: np.ndarray
    ) -> dict[CameraDirection, CameraData]:
        out: dict[CameraDirection, CameraData] = {}
        for position, (img_path, T_world_cam) in self._frame_cams[ts].items():
            direction = _WAYVE_CAM_TO_DIRECTION.get(position)
            if direction is None:
                continue
            cid = self._position_camera_id[position]
            K, dist = self._cam_calib[cid]
            T_ego_cam = (T_ego_world @ T_world_cam).astype(np.float32)
            out[direction] = CameraData(
                camera_direction=direction,
                image=_decode_rgb(img_path),
                intrinsics=K,
                extrinsics=T_ego_cam,
                distortion=dist,
                is_fisheye=True,
            )
        return out

    def _build_lidar(self, T_ego_world: np.ndarray) -> LidarData:
        """Transform the world-frame SfM cloud into the per-frame ego frame.

        ``T_ego_world`` must be the *world->ego* transform (``inv(T_world_ego)``);
        applying it to world points yields ego-frame points. (Passing the
        ego->world pose here instead is the classic sign-flip that makes the
        cloud drift opposite the true driving direction.)
        """
        pts = self._points_world_flu
        if len(pts) == 0:
            xyz_ego = np.zeros((0, 3), dtype=np.float32)
        else:
            homog = np.concatenate(
                [pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1
            )
            xyz_ego = (homog @ T_ego_world.T)[:, :3]
            within = np.linalg.norm(xyz_ego, axis=1) <= self.LIDAR_MAX_RANGE_M
            xyz_ego = xyz_ego[within].astype(np.float32)
        return LidarData(
            points=pd.DataFrame(xyz_ego, columns=[c.value for c in LidarComponent])
        )

    def _prepare_standardized_frame_data(self, raw_frame_data) -> StandardFrameData:
        scene_dir, ts = raw_frame_data
        scene_dir = Path(scene_dir)
        self._refresh_scene_cache(scene_dir)

        T_world_ego = self._ego_pose[ts]
        T_ego_world = np.linalg.inv(T_world_ego).astype(np.float32)
        x, y, z, heading = matrix_to_xyz_heading(T_world_ego)

        cameras = (
            self._build_cameras(ts, T_ego_world)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )
        lidar = (
            self._build_lidar(T_ego_world)
            if self.needs_attr(StandardFrameDataField.LIDAR)
            else None
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=scene_dir.name,
            frame_id=ts,
            timestamp=ts / 1e9,
            split=self._split,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [ts / 1e9],
                    TC.X: [x],
                    TC.Y: [y],
                    TC.Z: [z],
                    TC.HEADING: [heading],
                }
            ),
            cameras=cameras,
            lidar=lidar,
            aux_data={"pose_matrix": T_world_ego.astype(np.float32)},
            extra_index_data={"scene_id": scene_dir.name},
        )
