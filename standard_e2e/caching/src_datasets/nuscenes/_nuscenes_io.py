"""On-disk access for nuScenes, through nuscenes-devkit.

nuScenes ships as a normalized set of JSON tables under ``<dataroot>/<version>/``
joined by ``token``, plus ``samples/`` (the 2 Hz keyframe sensor files) and
``sweeps/`` (intermediate frames we don't use). The official ``nuscenes-devkit``
loads the tables into a :class:`~nuscenes.nuscenes.NuScenes` object and adds the
reverse indices this module reads: ``sample['data']`` (keyframe channel ->
``sample_data`` token), ``sample['anns']``, ``sample_data['sensor_modality']``
and ``sample_annotation['category_name']``.

A **frame** is one ``sample`` (keyframe); a **segment** is one ``scene``. Frames
are resolved into a compact, picklable :class:`NuscFrame` (paths + calibration
matrices + raw global boxes) so the converter can resolve them once in the parent
and hand them to worker processes without shipping any table state. Everything is
expressed in the **ego frame** of the sample's ``LIDAR_TOP`` capture: camera
``extrinsics`` are ``T_ego_from_camera`` and the lidar is moved into the ego frame
with ``T_ego_from_lidar`` (both from ``calibrated_sensor``); the ego pose itself
(``T_global_from_ego``) drives ``global_position`` and the past/future
trajectory. Cameras fire within a few ms of the lidar sweep, so reusing the
single lidar-time ego pose for all sensors is the standard nuScenes approximation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from standard_e2e.utils import quat_wxyz_to_rotmat, se3


@dataclass(frozen=True)
class NuscCamera:
    """One resolved camera view (no pixels yet -- ``image_path`` is read later).

    ``channel`` is the raw nuScenes channel (e.g. ``"CAM_FRONT"``); the processor
    maps it to a :class:`~standard_e2e.enums.CameraDirection`. ``extrinsics`` is
    ``T_ego_from_camera``; ``intrinsics`` is the pinhole ``camera_intrinsic``.
    """

    channel: str
    image_path: str
    intrinsics: np.ndarray  # (3, 3) float32
    extrinsics: np.ndarray  # (4, 4) float32, T_ego_from_camera


@dataclass(frozen=True)
class NuscBox:
    """One 3D annotation, carried in **global** coordinates (transformed to the
    ego frame by the processor). ``wlh`` is nuScenes' ``size`` order
    ``(width, length, height)``; ``rotation_wxyz`` is the Hamilton scalar-first
    global orientation quaternion."""

    category_name: str
    instance_token: str
    center_global: np.ndarray  # (3,) float64
    wlh: np.ndarray  # (3,) float64 (w, l, h)
    rotation_wxyz: np.ndarray  # (4,) float64


@dataclass(frozen=True)
class NuscFrame:
    """A fully-resolved, picklable nuScenes keyframe (one ``sample``)."""

    scene_name: str
    sample_token: str
    timestamp_s: float
    frame_id: int  # the sample timestamp in microseconds (unique, monotonic)
    pose_global_from_ego: np.ndarray  # (4, 4) float64, T_global_from_ego
    cameras: tuple[NuscCamera, ...]
    lidar_path: Optional[str]
    lidar_extrinsics: Optional[np.ndarray]  # (4, 4) float32, T_ego_from_lidar
    detections: tuple[NuscBox, ...]
    # Map location (e.g. ``"singapore-onenorth"``) and the resolved path to its
    # vector ``maps/expansion/<location>.json`` -- ``None`` when the map-expansion
    # pack is not present (then no HD map is built).
    location: Optional[str]
    map_expansion_path: Optional[str]


def load_nuscenes(dataroot: str, version: str) -> NuScenes:
    """Load the ``version`` metadata tables under ``dataroot`` with the devkit.

    The devkit holds every table in memory -- for ``v1.0-trainval`` that is
    ~2.6 M ``sample_data`` and ~1.2 M ``sample_annotation`` rows, several GB --
    so the converter loads it **once, in the parent process**. Worker processes
    never see it: they receive resolved :class:`NuscFrame` objects.
    """
    meta_dir = os.path.join(dataroot, version)
    if not os.path.isfile(os.path.join(meta_dir, "sample.json")):
        raise FileNotFoundError(
            f"nuScenes metadata not found at {meta_dir} (expected "
            f"{version}/sample.json). Download and extract the '{version}' "
            f"metadata + sensor blobs so <dataroot>/{version}/*.json and "
            f"<dataroot>/samples/ exist (e.g. tar -xf v1.0-mini.tgz)."
        )
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def map_expansion_path(dataroot: str, location: str) -> Optional[str]:
    """Path to ``maps/expansion/<location>.json`` (the vector map) for
    ``location``, or ``None`` if the map-expansion pack is not present."""
    path = os.path.join(dataroot, "maps", "expansion", f"{location}.json")
    return path if os.path.isfile(path) else None


def scene_sample_tokens(nusc: NuScenes, scene: dict[str, Any]) -> list[str]:
    """Sample tokens of ``scene`` in capture order (the ``next`` chain)."""
    tokens: list[str] = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def scene_has_sensor_data(nusc: NuScenes, scene: dict[str, Any]) -> bool:
    """Whether the scene's first keyframe ``LIDAR_TOP`` file exists on disk.

    A cheap proxy for "has this scene's sensor blob been downloaded?": the
    trainval ``*_blobs.tgz`` arrive one at a time, so the metadata lists all
    700 scenes long before most of their ``samples/`` files exist. The
    converter uses this to convert a partial download cleanly.
    """
    data = nusc.get("sample", scene["first_sample_token"])["data"]
    sd_token = data.get("LIDAR_TOP") or next(iter(data.values()), None)
    if sd_token is None:
        return False
    return os.path.isfile(nusc.get_sample_data_path(sd_token))


def _ego_pose_matrix(ego_pose: dict[str, Any]) -> np.ndarray:
    """``T_global_from_ego`` for an ``ego_pose`` record."""
    return se3(
        quat_wxyz_to_rotmat(ego_pose["rotation"]),
        ego_pose["translation"],
        dtype=np.float64,
    )


def _sensor_extrinsics(calib: dict[str, Any]) -> np.ndarray:
    """``T_ego_from_sensor`` for a ``calibrated_sensor`` record."""
    return se3(
        quat_wxyz_to_rotmat(calib["rotation"]), calib["translation"], dtype=np.float32
    )


def resolve_frame(nusc: NuScenes, sample_token: str) -> NuscFrame:
    """Build the compact :class:`NuscFrame` for one keyframe sample."""
    sample = nusc.get("sample", sample_token)
    data = sample["data"]

    # The ego pose at the lidar capture is the canonical frame pose; fall
    # back to any present sensor if a scene somehow lacks LIDAR_TOP.
    ref_sd_token = data.get("LIDAR_TOP") or next(iter(data.values()))
    ref_sd = nusc.get("sample_data", ref_sd_token)
    pose = _ego_pose_matrix(nusc.get("ego_pose", ref_sd["ego_pose_token"]))

    cameras: list[NuscCamera] = []
    lidar_path: Optional[str] = None
    lidar_extrinsics: Optional[np.ndarray] = None
    for channel, sd_token in sorted(data.items()):
        sd_record = nusc.get("sample_data", sd_token)
        calib = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        if sd_record["sensor_modality"] == "camera":
            cameras.append(
                NuscCamera(
                    channel=channel,
                    image_path=nusc.get_sample_data_path(sd_token),
                    intrinsics=np.asarray(
                        calib["camera_intrinsic"], dtype=np.float32
                    ).reshape(3, 3),
                    extrinsics=_sensor_extrinsics(calib),
                )
            )
        elif sd_record["sensor_modality"] == "lidar":
            lidar_path = nusc.get_sample_data_path(sd_token)
            lidar_extrinsics = _sensor_extrinsics(calib)
        # radar (and anything else) has no StandardE2E target -> skipped.

    detections: list[NuscBox] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        detections.append(
            NuscBox(
                category_name=ann["category_name"],
                instance_token=str(ann["instance_token"]),
                center_global=np.asarray(ann["translation"], dtype=np.float64),
                wlh=np.asarray(ann["size"], dtype=np.float64),
                rotation_wxyz=np.asarray(ann["rotation"], dtype=np.float64),
            )
        )

    scene = nusc.get("scene", sample["scene_token"])
    location = nusc.get("log", scene["log_token"])["location"]
    return NuscFrame(
        scene_name=scene["name"],
        sample_token=sample_token,
        timestamp_s=float(sample["timestamp"]) * 1e-6,
        frame_id=int(sample["timestamp"]),
        pose_global_from_ego=pose,
        cameras=tuple(cameras),
        lidar_path=lidar_path,
        lidar_extrinsics=lidar_extrinsics,
        detections=tuple(detections),
        location=location,
        map_expansion_path=map_expansion_path(nusc.dataroot, location),
    )


def iter_resolved_frames(nusc: NuScenes, scene_names: set[str]) -> Iterator[NuscFrame]:
    """Yield resolved frames for the scenes in ``scene_names``, scene-major and
    sample-time-ascending. Scenes absent from this ``version`` are skipped."""
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        for sample_token in scene_sample_tokens(nusc, scene):
            yield resolve_frame(nusc, sample_token)


def read_image(path: str) -> np.ndarray:
    """Read a camera JPEG as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read nuScenes image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
