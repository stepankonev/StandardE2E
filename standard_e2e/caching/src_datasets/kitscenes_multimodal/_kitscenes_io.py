"""On-disk access helpers for KITScenes Multimodal.

Each KITScenes scene is a directory named by a scene UUID::

    <scene_uuid>/
        calibration/calib.json          # all sensor intrinsics + extrinsics
        poses.txt                       # per-frame ego pose (TUM, map-local frame)
        timestamp.reference.txt         # per-frame reference timestamp (seconds)
        camera_ring_front/<frame>.jpg   # 6 surround "ring" cameras (+ base/stereo,
        camera_ring_front_left/ ...     #   not ingested), <frame> = 10-digit index
        lidar_top/<frame>.parquet       # 128-beam top LiDAR (+ 6 more, not ingested)
        lidar_*/, radar_*/, gnss*/, processed/, async/   # not ingested
        maps/map.osm + maps/origin.json # Lanelet2 HD map (-> HDMap)

``<frame>`` is a zero-padded 10-digit index (``0000000000`` ..); a scene is a
synchronized 10 Hz stream. The Hugging Face release groups scene tarballs by
split (``data/<split>/<scene_uuid>.tar``); after extraction this module resolves
the scene directories for a requested split (by the on-disk split folder), or
falls back to "every scene present" for a flat layout (e.g. the single-scene
sample). Framing / calibration math lives in :mod:`._kitscenes_geometry`; map
parsing in :mod:`._kitscenes_map`.

Only ``lidar_top`` and the six ``camera_ring_*`` views are read here (the scope
of this integration); the other LiDARs, the radars, the base/stereo cameras and
the GNSS/INS streams are present on disk but not ingested.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pyarrow.parquet as pq

from standard_e2e.caching.src_datasets.kitscenes_multimodal._kitscenes_splits import (
    SPLIT_DIR_ALIASES,
    split_dir_names,
)

# Per-frame file stem: 10-digit zero-padded index ("0000000000.jpg" ...).
_FRAME_STEM_WIDTH = 10

# Union of every on-disk split-folder name, used to detect whether a tree is
# organised by split at all (vs. a flat single-scene layout).
_ALL_SPLIT_DIR_NAMES: frozenset[str] = frozenset(
    name for names in SPLIT_DIR_ALIASES.values() for name in names
)

# How deep below ``--input_path`` we look for scene directories: the path itself,
# or scenes at ``<root>/<uuid>``, ``<root>/<split>/<uuid>`` or
# ``<root>/data/<split>/<uuid>`` (the HF layout).
_MAX_SCENE_SEARCH_DEPTH = 3


def _frame_stem(frame_index: int) -> str:
    return f"{frame_index:0{_FRAME_STEM_WIDTH}d}"


@dataclass(frozen=True)
class SceneRef:
    """Picklable locator for one KITScenes scene (one cache segment)."""

    scene_id: str  # scene UUID -> StandardFrameData.segment_id
    scene_dir: str  # absolute path to the scene directory

    @property
    def calib_path(self) -> str:
        return os.path.join(self.scene_dir, "calibration", "calib.json")

    @property
    def poses_path(self) -> str:
        return os.path.join(self.scene_dir, "poses.txt")

    @property
    def reference_timestamps_path(self) -> str:
        return os.path.join(self.scene_dir, "timestamp.reference.txt")

    @property
    def map_osm_path(self) -> str:
        return os.path.join(self.scene_dir, "maps", "map.osm")

    @property
    def map_origin_path(self) -> str:
        return os.path.join(self.scene_dir, "maps", "origin.json")

    def camera_path(self, camera_name: str, frame_index: int) -> str:
        return os.path.join(
            self.scene_dir, camera_name, f"{_frame_stem(frame_index)}.jpg"
        )

    def lidar_top_path(self, frame_index: int) -> str:
        return os.path.join(
            self.scene_dir, "lidar_top", f"{_frame_stem(frame_index)}.parquet"
        )


def _is_scene_dir(path: Path) -> bool:
    """A scene directory carries the per-scene calibration JSON."""
    return (path / "calibration" / "calib.json").is_file()


def _find_scene_dirs(root: Path) -> list[Path]:
    """All scene directories at or below ``root`` (depth <= search limit)."""
    found: set[Path] = set()
    if _is_scene_dir(root):
        found.add(root)
    pattern = ""
    for _ in range(_MAX_SCENE_SEARCH_DEPTH):
        pattern += "*/"
        for calib in root.glob(f"{pattern}calibration/calib.json"):
            found.add(calib.parent.parent)
    return sorted(found)


def _ancestor_names(root: Path, scene_dir: Path) -> set[str]:
    """Directory names strictly between ``root`` and ``scene_dir`` (exclusive of
    the scene UUID itself) -- used to detect the on-disk split folder."""
    try:
        rel_parts = scene_dir.relative_to(root).parts
    except ValueError:
        return set()
    return set(rel_parts[:-1])  # drop the scene-UUID leaf


def resolve_scene_dirs(input_path: str, split: str) -> list[Path]:
    """Resolve the scene directories to process for ``split`` under ``input_path``.

    If the tree is organised by split folder (the HF layout, or any tree with a
    ``train`` / ``validation`` / ``test`` / ... directory above the scenes), only
    scenes under the requested split's folder are returned. If no split folder is
    present (a flat layout, e.g. the single-scene sample), **every** scene found
    is returned and ``split`` is treated as a passthrough output label. ``split ==
    "all"`` always returns every scene found.

    Raises ``FileNotFoundError`` -- pointing at extraction -- when no scene
    directory (one with ``calibration/calib.json``) is found.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"KITScenes input path not found: {root}")

    scene_dirs = _find_scene_dirs(root)
    if not scene_dirs:
        raise FileNotFoundError(
            f"No KITScenes scene directories under {root} (expected a "
            f"<scene_uuid>/calibration/calib.json). Download and extract a split "
            f"first, e.g. scripts/prepare_dataset_kitscenes_multimodal.sh."
        )

    organised_by_split = any(
        _ancestor_names(root, d) & _ALL_SPLIT_DIR_NAMES for d in scene_dirs
    )
    if split == "all" or not organised_by_split:
        return scene_dirs

    wanted = set(split_dir_names(split))
    return [d for d in scene_dirs if _ancestor_names(root, d) & wanted]


def scene_frame_count(scene_dir: str) -> int:
    """Number of frames in a scene (the reference-timestamp count)."""
    ref = Path(scene_dir) / "timestamp.reference.txt"
    if not ref.is_file():
        raise FileNotFoundError(f"missing reference timestamps: {ref}")
    with open(ref, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def frames_for_scene_dirs(
    scene_dirs: list[Path],
) -> Iterator[tuple[SceneRef, int]]:
    """Yield ``(SceneRef, frame_index)`` for each scene, scene-major / ascending.

    Scene-major ordering keeps each segment's frames contiguous so the
    trajectory aggregator sees per-segment runs.
    """
    for scene_dir in scene_dirs:
        ref = SceneRef(scene_id=scene_dir.name, scene_dir=str(scene_dir))
        for frame_index in range(scene_frame_count(str(scene_dir))):
            yield ref, frame_index


def iter_scene_frames(input_path: str, split: str) -> Iterator[tuple[SceneRef, int]]:
    """``frames_for_scene_dirs`` over every scene resolved for ``split``."""
    return frames_for_scene_dirs(resolve_scene_dirs(input_path, split))


def read_calibration(path: str) -> dict:
    """Read ``calibration/calib.json`` (parsed by :mod:`._kitscenes_geometry`)."""
    with open(path, encoding="utf-8") as handle:
        calib: dict = json.load(handle)
    return calib


def read_poses(path: str) -> np.ndarray:
    """Read ``poses.txt`` -> ``(N, 8)`` float64 TUM rows.

    Columns are ``timestamp tx ty tz qx qy qz qw`` (quaternion scalar-last); the
    translation is the ego position in the Lanelet2 map-local frame.
    """
    poses = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if poses.size == 0:
        return np.zeros((0, 8), dtype=np.float64)
    if poses.shape[1] != 8:
        raise ValueError(
            f"{path}: expected 8 TUM columns (ts tx ty tz qx qy qz qw); "
            f"got {poses.shape[1]}"
        )
    return poses


def read_reference_timestamps(path: str) -> np.ndarray:
    """Read ``timestamp.reference.txt`` -> ``(N,)`` float64 seconds."""
    return np.loadtxt(path, dtype=np.float64, ndmin=1)


def read_lidar_top_xyz(path: str) -> np.ndarray:
    """Read a ``lidar_top/<frame>.parquet`` -> ``(N, 3)`` float32 xyz (sensor frame).

    KITScenes stores xyz as ``int32`` discretized by the schema-metadata
    ``discretization_resolution`` (meters/unit); invalid returns are flagged by
    ``reflectivity == -1`` with zero xyz and are dropped. StandardE2E keeps only
    xyz (the ``LidarComponent`` set); reflectivity / ring / timestamp columns are
    not retained.
    """
    table = pq.read_table(path, columns=["x", "y", "z", "reflectivity"])
    if table.num_rows == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    z = table.column("z").to_numpy()
    reflectivity = table.column("reflectivity").to_numpy()

    # Drop the invalid-return sentinel (reflectivity == -1 with zero xyz) before
    # de-discretizing, so phantom points don't pile up at the sensor origin.
    valid = ~((reflectivity == -1.0) & (x == 0) & (y == 0) & (z == 0))
    xyz = np.stack([x, y, z], axis=1)[valid].astype(np.float32)

    metadata = table.schema.metadata or {}
    resolution_key = b"discretization_resolution"
    if resolution_key in metadata:
        xyz *= np.float32(float(metadata[resolution_key]))
    return np.ascontiguousarray(xyz)


def read_image(path: str) -> np.ndarray:
    """Read a camera JPEG as a contiguous ``(H, W, 3)`` uint8 RGB array."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
