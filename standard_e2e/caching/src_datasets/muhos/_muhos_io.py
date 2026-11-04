"""On-disk access helpers for the MUHOS dataset (M3S Kuura, University of Oulu).

MUHOS ships one ``.tar`` per sensor stream and driving set (see
``datasets/muhos/README.md``); each archive holds one folder, and extracting all
of a set's archives into one directory gives::

    <root>/<set>/                                  # e.g. imu_highway_processed
        aux_image_color[_anonymized]/image_<sec>_<nsec>.png   # color, 25 Hz
        left_image_rect[_anonymized]/image_<sec>_<nsec>.png   # stereo, rectified
        right_image_rect[_anonymized]/image_<sec>_<nsec>.png
        left_depth/  left_cost/                    # stereo products (not read)
        lidar_points/pointcloud_<sec>_<nsec>.laz   # Hesai OT128, 10 Hz
        can/toyota_CAN_<...>.json                  # vehicle bus
        can/toyota_GPS_<...>.json                  # PCAN-GPS
        imu/lidar_imu.json                         # the lidar's IMU (sensor clock)

A *set* (one continuous drive) is the StandardE2E **segment**; it is discovered by its
``lidar_points/`` and ``can/`` folders. Camera folders carry an ``_anonymized``
suffix in most sets (faces and plates blurred) and none in the two gravel sets;
both are accepted. File names carry the ROS header stamp; the color image and
the stereo pair of one capture share it.

* **lidar**: ``.laz`` (LAS 1.2, point format 3) with ``x, y, z`` in metres
  (scale 1 mm) in the lidar frame, ``intensity`` and a per-point time in
  ``gps_time``, which the export wrote as Unix time minus 315964800 s (no
  leap-second correction). A sweep is 128 x 1800 points, including ``(0, 0, 0)``
  for missing returns; the file's stamp is the start of the 100 ms sweep.
* **CAN / GPS**: lists of decoded signals ``{timestamp, frame_id, message,
  signal, value}``, timestamped at reception (Unix seconds) by the bus logger, whose
  clock is offset from the sensors' by 0.3-1.3 s and drifts (see ``_muhos_ego``).
  :func:`read_signal` extracts one signal.
* **lidar IMU**: ``imu/lidar_imu.json``, the lidar's 100 Hz IMU on the sensor clock,
  written as several concatenated JSON arrays. :func:`read_imu` reads it.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import laspy
import numpy as np

# The export stored per-point times as GPS seconds = Unix seconds - this offset.
GPS_UNIX_OFFSET_S = 315964800.0

_STAMP_PATTERN = re.compile(r"_(\d+)_(\d+)\.(?:png|laz)$")
_CAMERA_FOLDERS = {
    "aux": ("aux_image_color_anonymized", "aux_image_color"),
    "left": ("left_image_rect_anonymized", "left_image_rect"),
    "right": ("right_image_rect_anonymized", "right_image_rect"),
}


@dataclass(frozen=True)
class SetRef:
    """One MUHOS driving set (a StandardE2E segment) on disk."""

    set_id: str
    path: str
    can_json: str
    gps_json: str
    imu_json: str | None = None  # imu/lidar_imu.json

    def camera_dir(self, camera: str) -> str | None:
        for name in _CAMERA_FOLDERS[camera]:
            path = os.path.join(self.path, name)
            if os.path.isdir(path):
                return path
        return None

    @property
    def lidar_dir(self) -> str:
        return os.path.join(self.path, "lidar_points")


def set_id_from_folder(folder_name: str) -> str:
    """``imu_highway_processed`` -> ``highway`` (other names are kept as they are)."""
    name = folder_name
    if name.startswith("imu_"):
        name = name[len("imu_") :]
    if name.endswith("_processed"):
        name = name[: -len("_processed")]
    return name


def discover_sets(input_path: str) -> list[SetRef]:
    """Driving sets under ``input_path`` (or ``input_path`` itself), sorted by id."""
    candidates = [Path(input_path)] + sorted(
        p for p in Path(input_path).iterdir() if p.is_dir()
    )
    sets = []
    for path in candidates:
        if not (path / "lidar_points").is_dir() or not (path / "can").is_dir():
            continue
        can_files = sorted(glob.glob(str(path / "can" / "toyota_CAN_*.json")))
        gps_files = sorted(glob.glob(str(path / "can" / "toyota_GPS_*.json")))
        if len(can_files) != 1 or len(gps_files) != 1:
            raise ValueError(
                f"MUHOS set {path} should hold one toyota_CAN_*.json and one "
                f"toyota_GPS_*.json in can/; found {len(can_files)} and "
                f"{len(gps_files)}."
            )
        imu = path / "imu" / "lidar_imu.json"
        sets.append(
            SetRef(
                set_id_from_folder(path.name),
                str(path),
                can_files[0],
                gps_files[0],
                str(imu) if imu.is_file() else None,
            )
        )
    return sorted(sets, key=lambda s: s.set_id)


def list_stamped_files(directory: str | None) -> tuple[np.ndarray, list[str]]:
    """``(stamps_ns, paths)`` of a stream folder, sorted by the stamp in the file
    name."""
    if directory is None or not os.path.isdir(directory):
        return np.zeros((0,), dtype=np.int64), []
    rows = []
    for entry in os.scandir(directory):
        match = _STAMP_PATTERN.search(entry.name)
        if match:
            rows.append(
                (int(match.group(1)) * 1_000_000_000 + int(match.group(2)), entry.path)
            )
    rows.sort()
    return np.array([r[0] for r in rows], dtype=np.int64), [r[1] for r in rows]


def read_signal(rows: list[dict], message: str, signal: str) -> np.ndarray:
    """``(N, 2)`` float64 ``[timestamp_s, value]`` of one decoded signal,
    time-sorted."""
    out = np.array(
        [
            (r["timestamp"], r["value"])
            for r in rows
            if r["message"] == message and r["signal"] == signal
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    return out[np.argsort(out[:, 0], kind="stable")]


def read_json(path: str) -> list[dict]:
    with open(path) as fh:
        rows: list[dict] = json.load(fh)
    return rows


def read_imu(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t_s (N,), angular_velocity (N, 3) rad/s)`` of a MUHOS IMU log, time-sorted.
    The file holds several concatenated JSON arrays (one per recording chunk)."""
    with open(path) as fh:
        text = fh.read()
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    i = 0
    while True:
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text):
            break
        obj, i = decoder.raw_decode(text, i)
        rows.extend(obj if isinstance(obj, list) else [obj])
    t = np.array([r["timestamp_ns"] for r in rows], dtype=np.float64) / 1e9
    w = np.array(
        [[r["angular_velocity"][k] for k in "xyz"] for r in rows], dtype=np.float64
    ).reshape(-1, 3)
    order = np.argsort(t, kind="stable")
    return t[order], w[order]


def read_lidar(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(xyz (N, 3) float64 lidar frame, intensity (N,) float32, unix_time (N,)
    float64)``, missing returns (exactly ``(0, 0, 0)``) removed."""
    las = laspy.read(path)
    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1)
    keep = np.any(xyz != 0.0, axis=1)
    intensity = np.asarray(las.intensity, dtype=np.float32)
    unix_time = np.asarray(las.gps_time, dtype=np.float64) + GPS_UNIX_OFFSET_S
    return xyz[keep], intensity[keep], unix_time[keep]


def read_image(path: str) -> np.ndarray:
    """RGB ``uint8`` ``(H, W, 3)``."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read MUHOS image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
