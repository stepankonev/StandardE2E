"""On-disk access for nuScenes -- read the JSON tables directly, no devkit.

nuScenes ships as a normalized set of JSON tables under ``<dataroot>/<version>/``
joined by ``token``, plus ``samples/`` (the 2 Hz keyframe sensor files) and
``sweeps/`` (intermediate frames we don't use). The official ``nuscenes-devkit``
loads these into a ``NuScenes`` object and builds a handful of reverse indices;
we can't depend on it (it pins ``numpy<2`` / ``Shapely~=2.0.3`` against this
project's numpy 2.x), so this module reads the tables and rebuilds exactly the
indices we need, matching the devkit's ``__make_reverse_index__``:

* ``sample -> {channel: sample_data_token}`` for keyframes and
  ``sample -> [sample_annotation_token]`` (devkit's ``sample['data']`` /
  ``sample['anns']``);
* a ``sample_data``'s sensor ``channel`` / ``modality`` via
  ``calibrated_sensor -> sensor``;
* a ``sample_annotation``'s ``category_name`` via ``instance -> category``
  (it is **not** stored on the annotation record).

The clone under ``se2e_extarnal_apis/nuscenes-devkit`` is the schema reference.

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

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator, Optional, cast

import cv2
import numpy as np

from standard_e2e.utils import quat_wxyz_to_rotmat, se3

# nuScenes raw lidar is packed float32 with five columns per point
# (x, y, z, intensity, ring_index); StandardE2E keeps xyz.
_LIDAR_NUM_COLS = 5

# Tables we read (a subset of the full schema -- enough to rebuild the frame).
_REQUIRED_TABLES = (
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "calibrated_sensor",
    "ego_pose",
    "sensor",
    "instance",
    "category",
    "log",
)


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


def _load_table(meta_dir: str, name: str) -> list[dict[str, Any]]:
    with open(os.path.join(meta_dir, f"{name}.json"), encoding="utf-8") as fh:
        return cast("list[dict[str, Any]]", json.load(fh))


class NuscTables:
    """Loads the nuScenes JSON tables for one ``version`` and resolves frames.

    Instantiated **once in the parent process** (the converter): for the full
    ``v1.0-trainval`` the ``sample_annotation`` table is ~1.1 M rows, so this
    holds a few GB. Worker processes never build this -- they receive resolved
    :class:`NuscFrame` objects -- so that cost is paid once, not per worker.
    """

    def __init__(self, dataroot: str, version: str) -> None:
        self.dataroot = dataroot
        self.version = version
        meta_dir = os.path.join(dataroot, version)
        if not os.path.isfile(os.path.join(meta_dir, "sample.json")):
            raise FileNotFoundError(
                f"nuScenes metadata not found at {meta_dir} (expected "
                f"{version}/sample.json). Download and extract the '{version}' "
                f"metadata + sensor blobs so <dataroot>/{version}/*.json and "
                f"<dataroot>/samples/ exist (e.g. tar -xf v1.0-mini.tgz)."
            )

        self.scenes = _load_table(meta_dir, "scene")
        self._scene_name: dict[str, str] = {s["token"]: s["name"] for s in self.scenes}
        logs = {log["token"]: log for log in _load_table(meta_dir, "log")}
        # scene_token -> map location (via its log), used to find the vector map.
        self._scene_location: dict[str, str] = {
            s["token"]: logs[s["log_token"]]["location"] for s in self.scenes
        }
        self._map_path_cache: dict[str, Optional[str]] = {}
        self._sample = {r["token"]: r for r in _load_table(meta_dir, "sample")}
        self._calib = {
            r["token"]: r for r in _load_table(meta_dir, "calibrated_sensor")
        }
        self._ego = {r["token"]: r for r in _load_table(meta_dir, "ego_pose")}
        self._sensor = {r["token"]: r for r in _load_table(meta_dir, "sensor")}
        self._instance = {r["token"]: r for r in _load_table(meta_dir, "instance")}
        self._category = {r["token"]: r for r in _load_table(meta_dir, "category")}

        # Only keyframe sample_data are ever looked up (the ~2.2 M intermediate
        # sweep records are unused), so keep just those -- on v1.0-trainval that
        # is ~410 k of 2.6 M rows, cutting the parent's peak RAM and load time.
        self._sd: dict[str, dict[str, Any]] = {}
        # sample_token -> {channel: sample_data_token} (devkit's ``sample['data']``).
        self._keyframe_data: dict[str, dict[str, str]] = defaultdict(dict)
        for record in _load_table(meta_dir, "sample_data"):
            if not record["is_key_frame"]:
                continue
            self._sd[record["token"]] = record
            channel = self._channel_of(record)
            self._keyframe_data[record["sample_token"]][channel] = record["token"]

        annotations = _load_table(meta_dir, "sample_annotation")
        self._ann = {r["token"]: r for r in annotations}
        # sample_token -> [annotation_token] (devkit's ``sample['anns']``).
        self._sample_anns: dict[str, list[str]] = defaultdict(list)
        for record in annotations:
            self._sample_anns[record["sample_token"]].append(record["token"])

    # --- reverse-index helpers (mirror the devkit) -------------------------

    def _sensor_of(self, sd_record: dict[str, Any]) -> dict[str, Any]:
        calib = self._calib[sd_record["calibrated_sensor_token"]]
        return self._sensor[calib["sensor_token"]]

    def _channel_of(self, sd_record: dict[str, Any]) -> str:
        return str(self._sensor_of(sd_record)["channel"])

    def _category_name(self, instance_token: str) -> str:
        category_token = self._instance[instance_token]["category_token"]
        return str(self._category[category_token]["name"])

    def map_expansion_path(self, location: str) -> Optional[str]:
        """Path to ``maps/expansion/<location>.json`` (the vector map) for
        ``location``, or ``None`` if the map-expansion pack is not present."""
        if location not in self._map_path_cache:
            path = os.path.join(self.dataroot, "maps", "expansion", f"{location}.json")
            self._map_path_cache[location] = path if os.path.isfile(path) else None
        return self._map_path_cache[location]

    # --- enumeration -------------------------------------------------------

    def scene_sample_tokens(self, scene: dict[str, Any]) -> list[str]:
        """Sample tokens of ``scene`` in capture order (the ``next`` chain)."""
        tokens: list[str] = []
        token = scene["first_sample_token"]
        while token:
            tokens.append(token)
            token = self._sample[token]["next"]
        return tokens

    def scene_has_sensor_data(self, scene: dict[str, Any]) -> bool:
        """Whether the scene's first keyframe ``LIDAR_TOP`` file exists on disk.

        A cheap proxy for "has this scene's sensor blob been downloaded?": the
        trainval ``*_blobs.tgz`` arrive one at a time, so the metadata lists all
        700 scenes long before most of their ``samples/`` files exist. The
        converter uses this to convert a partial download cleanly.
        """
        data = self._keyframe_data.get(scene["first_sample_token"], {})
        sd_token = data.get("LIDAR_TOP") or (
            next(iter(data.values())) if data else None
        )
        if sd_token is None:
            return False
        return os.path.isfile(
            os.path.join(self.dataroot, self._sd[sd_token]["filename"])
        )

    # --- per-frame resolution ----------------------------------------------

    def _ego_pose_matrix(self, ego_pose_token: str) -> np.ndarray:
        ego = self._ego[ego_pose_token]
        return se3(
            quat_wxyz_to_rotmat(ego["rotation"]), ego["translation"], dtype=np.float64
        )

    def _sensor_extrinsics(self, sd_record: dict[str, Any]) -> np.ndarray:
        """``T_ego_from_sensor`` for a sample_data, from its calibrated_sensor."""
        calib = self._calib[sd_record["calibrated_sensor_token"]]
        return se3(
            quat_wxyz_to_rotmat(calib["rotation"]),
            calib["translation"],
            dtype=np.float32,
        )

    def resolve_frame(self, sample_token: str) -> NuscFrame:
        """Build the compact :class:`NuscFrame` for one keyframe sample."""
        sample = self._sample[sample_token]
        data = self._keyframe_data[sample_token]

        # The ego pose at the lidar capture is the canonical frame pose; fall
        # back to any present sensor if a scene somehow lacks LIDAR_TOP.
        lidar_sd_token = data.get("LIDAR_TOP")
        ref_sd_token = lidar_sd_token or next(iter(data.values()))
        pose = self._ego_pose_matrix(self._sd[ref_sd_token]["ego_pose_token"])

        cameras: list[NuscCamera] = []
        lidar_path: Optional[str] = None
        lidar_extrinsics: Optional[np.ndarray] = None
        for channel, sd_token in sorted(data.items()):
            sd_record = self._sd[sd_token]
            modality = str(self._sensor_of(sd_record)["modality"])
            if modality == "camera":
                calib = self._calib[sd_record["calibrated_sensor_token"]]
                cameras.append(
                    NuscCamera(
                        channel=channel,
                        image_path=os.path.join(self.dataroot, sd_record["filename"]),
                        intrinsics=np.asarray(
                            calib["camera_intrinsic"], dtype=np.float32
                        ).reshape(3, 3),
                        extrinsics=self._sensor_extrinsics(sd_record),
                    )
                )
            elif modality == "lidar":
                lidar_path = os.path.join(self.dataroot, sd_record["filename"])
                lidar_extrinsics = self._sensor_extrinsics(sd_record)
            # radar (and anything else) has no StandardE2E target -> skipped.

        detections: list[NuscBox] = []
        for ann_token in self._sample_anns.get(sample_token, []):
            ann = self._ann[ann_token]
            detections.append(
                NuscBox(
                    category_name=self._category_name(ann["instance_token"]),
                    instance_token=str(ann["instance_token"]),
                    center_global=np.asarray(ann["translation"], dtype=np.float64),
                    wlh=np.asarray(ann["size"], dtype=np.float64),
                    rotation_wxyz=np.asarray(ann["rotation"], dtype=np.float64),
                )
            )

        location = self._scene_location.get(sample["scene_token"])
        map_path = self.map_expansion_path(location) if location else None
        return NuscFrame(
            scene_name=self._scene_name[sample["scene_token"]],
            sample_token=sample_token,
            timestamp_s=float(sample["timestamp"]) * 1e-6,
            frame_id=int(sample["timestamp"]),
            pose_global_from_ego=pose,
            cameras=tuple(cameras),
            lidar_path=lidar_path,
            lidar_extrinsics=lidar_extrinsics,
            detections=tuple(detections),
            location=location,
            map_expansion_path=map_path,
        )


def iter_resolved_frames(
    tables: NuscTables, scene_names: set[str]
) -> Iterator[NuscFrame]:
    """Yield resolved frames for the scenes in ``scene_names``, scene-major and
    sample-time-ascending. Scenes absent from this ``version`` are skipped."""
    for scene in tables.scenes:
        if scene["name"] not in scene_names:
            continue
        for sample_token in tables.scene_sample_tokens(scene):
            yield tables.resolve_frame(sample_token)


def read_image(path: str) -> np.ndarray:
    """Read a camera JPEG as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read nuScenes image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def read_lidar_xyz(path: str) -> np.ndarray:
    """Read a LIDAR_TOP ``.pcd.bin`` and return its ``(N, 3)`` float32 xyz.

    The file is packed ``float32`` with five columns per point
    (x, y, z, intensity, ring_index); StandardE2E keeps xyz. Points are in the
    LIDAR_TOP sensor frame and must be moved into the ego frame by the caller.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if raw.size % _LIDAR_NUM_COLS != 0:
        raise ValueError(
            f"{path}: {raw.size} float32 values not divisible by "
            f"{_LIDAR_NUM_COLS} lidar columns"
        )
    return np.ascontiguousarray(raw.reshape(-1, _LIDAR_NUM_COLS)[:, :3])
