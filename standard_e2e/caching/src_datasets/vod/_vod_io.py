"""On-disk access helpers for View-of-Delft (extracted detection release).

VoD ships as password-protected zips that **must be extracted first**
(``scripts/extract_vod.sh``), reproducing the official KITTI-style layout::

    <root>/lidar/
        ImageSets/{train,val,test}.txt
        training/{calib,velodyne,image_2,label_2,pose}/<frame>.{txt,bin,jpg,txt,json}
        testing/{calib,velodyne,image_2,pose}/<frame>.{...}     # no label_2/

``<frame>`` is a zero-padded 5-digit global index (``00000`` .. ``09930``). Only
the ``lidar/`` tree is read here: it carries the Velodyne HDL-64 cloud, the front
camera, the KITTI 3D labels and the velodyne<-camera calibration. The parallel
``radar*`` trees hold the 3+1D radar (no StandardE2E modality yet) and are not
ingested. Train and val frames live under ``training/`` (labelled); test frames
under ``testing/`` (sensor-only). This module resolves the dataset root, buckets
present frames into scenes per split, and reads the per-frame raw files; framing
math lives in :mod:`._vod_geometry`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from standard_e2e.caching.src_datasets.vod._vod_splits import VodScene, scenes_for_split

# Velodyne HDL-64 cloud: float32, 4 columns (x, y, z, reflectance). StandardE2E
# keeps only xyz (the ``LidarComponent`` set); reflectance is dropped.
_VELODYNE_NUM_COLS = 4

# KITTI label fields we use (0-based): 0 class, 1 truncation-or-track-id,
# 2 occluded, 8:11 dimensions (h, w, l), 11:14 location (x, y, z) in camera,
# 14 rotation about LiDAR -Z, 15 score (optional).
_LABEL_MIN_FIELDS = 15


@dataclass(frozen=True)
class FrameRef:
    """Picklable locator for one VoD frame (one keyframe of one scene)."""

    root: str  # directory that contains the ``lidar/`` tree
    scene_name: str  # e.g. "delft_2" -> StandardFrameData.segment_id
    subdir: str  # "training" | "testing"
    frame_id: int  # global frame index
    split: str  # "train" | "val" | "test"

    @property
    def stem(self) -> str:
        return f"{self.frame_id:05d}"

    def _sensor_path(self, kind: str, ext: str) -> str:
        return os.path.join(self.root, "lidar", self.subdir, kind, f"{self.stem}.{ext}")

    @property
    def calib_path(self) -> str:
        return self._sensor_path("calib", "txt")

    @property
    def pose_path(self) -> str:
        return self._sensor_path("pose", "json")

    @property
    def velodyne_path(self) -> str:
        return self._sensor_path("velodyne", "bin")

    @property
    def image_path(self) -> str:
        return self._sensor_path("image_2", "jpg")

    @property
    def label_path(self) -> str:
        return self._sensor_path("label_2", "txt")


@dataclass
class VodLabel:
    """One parsed KITTI label line (camera-frame box + metadata)."""

    cls: str
    track_field: str  # field 1: truncation (base labels) or track id (track-id set)
    occluded: int
    location_cam: np.ndarray  # (3,) camera-frame center (x, y, z)
    dimensions_hwl: np.ndarray  # (3,) (height, width, length)
    rotation: float  # yaw about LiDAR -Z
    score: float


def resolve_root(input_path: str) -> str:
    """Return the directory that holds the ``lidar/`` tree.

    Accepts ``--input_path`` pointing at the extracted root, at the
    ``view_of_delft_PUBLIC`` folder, or at its parent. Raises ``FileNotFoundError``
    -- pointing at the extraction step -- when no ``lidar/`` tree is found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"VoD input path not found: {root}")

    candidates = [root, root / "view_of_delft_PUBLIC", *root.glob("*/")]
    for candidate in candidates:
        if (candidate / "lidar" / "training" / "velodyne").is_dir() or (
            candidate / "lidar" / "testing" / "velodyne"
        ).is_dir():
            return str(candidate)

    raise FileNotFoundError(
        f"No VoD 'lidar/' tree under {root} (expected "
        f"lidar/training/velodyne or lidar/testing/velodyne). Extract the "
        f"detection zip first, e.g. scripts/extract_vod.sh."
    )


def _present_frame_ids(root: str, subdir: str) -> set[int]:
    """Integer stems of the ``*.bin`` clouds present under ``lidar/<subdir>``."""
    velodyne_dir = Path(root) / "lidar" / subdir / "velodyne"
    if not velodyne_dir.is_dir():
        return set()
    ids: set[int] = set()
    for entry in velodyne_dir.iterdir():
        if entry.suffix == ".bin" and entry.stem.isdigit():
            ids.add(int(entry.stem))
    return ids


def frame_refs_for_scenes(root: str, scenes: list[VodScene]) -> Iterator[FrameRef]:
    """Yield ``FrameRef`` for every present frame of ``scenes``, scene-major.

    Frames are ordered scene-major / frame-ascending so the trajectory aggregator
    sees contiguous per-scene runs. Frames listed in a scene range but absent on
    disk (dropped from the public release) are silently skipped. Each frame's
    split label is taken from its scene.
    """
    present_by_subdir: dict[str, set[int]] = {}
    for scene in scenes:
        present = present_by_subdir.get(scene.subdir)
        if present is None:
            present = _present_frame_ids(root, scene.subdir)
            present_by_subdir[scene.subdir] = present
        for frame_id in sorted(f for f in present if scene.contains(f)):
            yield FrameRef(root, scene.name, scene.subdir, frame_id, scene.split)


def iter_frame_refs(root: str, split: str) -> Iterator[FrameRef]:
    """``frame_refs_for_scenes`` over every scene assigned to ``split``."""
    return frame_refs_for_scenes(root, scenes_for_split(split))


def read_calibration_text(path: str) -> str:
    """Read a ``calib/<id>.txt`` body (parsed by :func:`._vod_geometry`)."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_pose(path: str) -> dict[str, np.ndarray]:
    """Read ``pose/<id>.json`` -> ``{name: (4, 4) float64}``.

    The file holds three single-line JSON objects (``odomToCamera``,
    ``mapToCamera``, ``UTMToCamera``), each a 16-value row-major 4x4. Parsing is
    keyed (not positional) so object order does not matter.
    """
    matrices: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            for key, values in json.loads(line).items():
                matrices[key] = np.asarray(values, dtype=np.float64).reshape(4, 4)
    return matrices


def read_velodyne_xyz(path: str) -> np.ndarray:
    """Read a Velodyne ``.bin`` and return its ``(N, 3)`` float32 xyz columns.

    The binary is float32 with 4 columns per point (x, y, z, reflectance);
    StandardE2E keeps only xyz. Points are already in the velodyne (ego) frame.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if raw.size % _VELODYNE_NUM_COLS != 0:
        raise ValueError(
            f"{path}: {raw.size} float32 values not divisible by "
            f"{_VELODYNE_NUM_COLS} velodyne columns"
        )
    return np.ascontiguousarray(raw.reshape(-1, _VELODYNE_NUM_COLS)[:, :3])


def read_image(path: str) -> np.ndarray:
    """Read a camera JPEG as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def read_label_lines(path: str) -> list[str]:
    """Non-empty lines of ``label_2/<id>.txt`` (``[]`` if absent, e.g. test)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def parse_label_line(line: str) -> VodLabel | None:
    """Parse one KITTI label line into a :class:`VodLabel` (``None`` if malformed).

    Field 1 is the KITTI truncation slot, which VoD overloads to carry the track
    id when the ``label_2_with_track_ids`` set is installed; it is preserved
    verbatim as ``track_field``.
    """
    parts = line.split()
    if len(parts) < _LABEL_MIN_FIELDS:
        return None
    return VodLabel(
        cls=parts[0],
        track_field=parts[1],
        occluded=int(float(parts[2])),
        dimensions_hwl=np.array(parts[8:11], dtype=np.float64),
        location_cam=np.array(parts[11:14], dtype=np.float64),
        rotation=float(parts[14]),
        score=float(parts[15]) if len(parts) > 15 else 1.0,
    )
