"""Minimal, dependency-free reader for COLMAP binary reconstructions.

WayveScenes101 ships a per-scene COLMAP ``rig`` reconstruction
(``cameras.bin``, ``images.bin``, ``points3D.bin``). The official SDK loads
it with ``pycolmap``, which materialises the full 2D feature tracks —
``images.bin`` is ~100 MB of per-image 2D points we never use. This reader
parses only what the preprocessing pipeline needs (camera intrinsics,
per-image world poses, and the 3D points), **seeking past** the 2D-point and
track blobs, so it is both dependency-free and much faster on the hot path.

Binary format reference: https://colmap.github.io/format.html#binary-file-format

Coordinate conventions are COLMAP's: cameras use OpenCV / RDF (x-right,
y-down, z-forward) and ``images.bin`` stores ``cam_from_world`` (world → camera)
as a (qw, qx, qy, qz) quaternion plus translation. Callers convert to the
Wayve FLU frame separately (see the dataset processor).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COLMAP camera model id -> number of params. We only need the count to read
# the right number of float64s; the meaning of params[:4] = (fx, fy, cx, cy)
# holds for every model WayveScenes uses (OPENCV_FISHEYE = id 5, 8 params).
_MODEL_NUM_PARAMS: dict[int, int] = {
    0: 3,  # SIMPLE_PINHOLE
    1: 4,  # PINHOLE
    2: 4,  # SIMPLE_RADIAL
    3: 5,  # RADIAL
    4: 8,  # OPENCV
    5: 8,  # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,  # FOV
    8: 4,  # SIMPLE_RADIAL_FISHEYE
    9: 5,  # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


@dataclass
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray  # float64, (num_params,); [:4] = fx, fy, cx, cy


@dataclass
class ColmapImage:
    image_id: int
    qvec_wxyz: np.ndarray  # float64 (4,), cam_from_world rotation (qw, qx, qy, qz)
    tvec: np.ndarray  # float64 (3,), cam_from_world translation
    camera_id: int
    name: str  # e.g. "front-forward/<timestamp_ns>.jpeg"


def _read(fp, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fp.read(size))


def read_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "rb") as fp:
        (num_cameras,) = _read(fp, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fp, "<iiQQ")
            n = _MODEL_NUM_PARAMS.get(model_id)
            if n is None:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            params = np.array(_read(fp, "<" + "d" * n), dtype=np.float64)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model_id=model_id,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_bin(path: Path) -> list[ColmapImage]:
    """Read per-image poses + names, skipping the 2D-point blobs."""
    images: list[ColmapImage] = []
    with open(path, "rb") as fp:
        (num_images,) = _read(fp, "<Q")
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(fp, "<idddddddi")
            # name: null-terminated bytes
            name_bytes = bytearray()
            while True:
                ch = fp.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes += ch
            (num_points2D,) = _read(fp, "<Q")
            # Skip the 2D points: each is (float64 x, float64 y, uint64 id) = 24 B.
            fp.seek(num_points2D * 24, 1)
            images.append(
                ColmapImage(
                    image_id=image_id,
                    qvec_wxyz=np.array([qw, qx, qy, qz], dtype=np.float64),
                    tvec=np.array([tx, ty, tz], dtype=np.float64),
                    camera_id=camera_id,
                    name=name_bytes.decode("utf-8"),
                )
            )
    return images


def read_points3D_bin(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (N,3) float64, rgb (N,3) uint8, error (N,) float64,
    track_length (N,) int64), skipping per-track detail.
    """
    xyz_list: list[tuple[float, float, float]] = []
    rgb_list: list[tuple[int, int, int]] = []
    err_list: list[float] = []
    track_list: list[int] = []
    with open(path, "rb") as fp:
        (num_points,) = _read(fp, "<Q")
        for _ in range(num_points):
            # uint64 id, 3x float64 xyz, 3x uint8 rgb, float64 error, uint64 track_len
            _pid, x, y, z, r, g, b, err, track_len = _read(fp, "<QdddBBBdQ")
            xyz_list.append((x, y, z))
            rgb_list.append((r, g, b))
            err_list.append(err)
            track_list.append(track_len)
            # Skip the track: track_len x (uint32 image_id, uint32 point2D_idx) = 8 B.
            fp.seek(track_len * 8, 1)
    xyz = np.array(xyz_list, dtype=np.float64).reshape(-1, 3)
    rgb = np.array(rgb_list, dtype=np.uint8).reshape(-1, 3)
    error = np.array(err_list, dtype=np.float64).reshape(-1)
    track_length = np.array(track_list, dtype=np.int64).reshape(-1)
    return xyz, rgb, error, track_length


def qvec_wxyz_to_rotmat(qvec_wxyz: np.ndarray) -> np.ndarray:
    """COLMAP (qw, qx, qy, qz) -> 3x3 rotation matrix (cam_from_world)."""
    w, x, y, z = qvec_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
