"""Source-dataset converter for MUHOS.

Discovers the driving sets (see :mod:`._muhos_io`), builds each set's ego trajectory
once in the parent process (:mod:`._muhos_ego`: the CAN and GPS logs are read in full)
and yields one :class:`FrameRef` per emitted frame, set by set in time order.

A **frame** is one lidar sweep (10 Hz) paired with the colour image taken during it --
the image whose capture time (its stamp minus
:data:`._muhos_calibration.CAMERA_DELAY_S`) falls inside the sweep's 100 ms and lies
closest to its middle; the frame's timestamp is that capture time, and the processor
moves the sweep's points to it. Sweeps without such an image are skipped (and counted
in the log). The ``FrameRef`` carries the pose at the frame time plus a short slice of
the trajectory around it for that motion compensation, so workers never read the
CAN/GPS logs.
With ``STANDARD_E2E_DEBUG=true`` only the first set is processed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.muhos import _muhos_calibration as calib
from standard_e2e.caching.src_datasets.muhos import _muhos_io as io
from standard_e2e.caching.src_datasets.muhos._muhos_ego import (
    EgoTrajectory,
    build_ego_trajectory,
)

SWEEP_DURATION_S = 0.1
# Trajectory samples shipped with each frame to motion-compensate its sweep.
_SLICE_HALF_WIDTH_S = 0.15


@dataclass(frozen=True)
class FrameRef:
    set_id: str
    frame_index: int
    lidar_path: str
    lidar_stamp_ns: int
    image_path: str
    image_stamp_ns: int
    timestamp: float  # frame time = image capture time (stamp - CAMERA_DELAY_S), s
    pose: tuple[float, float, float, float]  # x, y, heading, speed at ``timestamp``
    trajectory_slice: np.ndarray  # (N, 4): t, x, y, heading around ``timestamp``
    anchor_lat: float
    anchor_lon: float


def match_images_to_sweeps(sweep_ns: np.ndarray, image_ns: np.ndarray) -> np.ndarray:
    """For each sweep, the index of the image taken during it closest to its middle,
    or -1."""
    out = np.full(len(sweep_ns), -1, dtype=np.int64)
    if len(image_ns) == 0:
        return out
    duration_ns = int(SWEEP_DURATION_S * 1e9)
    lo = np.searchsorted(image_ns, sweep_ns, side="left")
    hi = np.searchsorted(image_ns, sweep_ns + duration_ns, side="left")
    middle = sweep_ns + duration_ns // 2
    for k in np.flatnonzero(hi > lo):
        cand = np.arange(lo[k], hi[k])
        out[k] = cand[np.argmin(np.abs(image_ns[cand] - middle[k]))]
    return out


def iter_set_frames(ref: io.SetRef, ego: EgoTrajectory) -> Iterator[FrameRef]:
    sweep_ns, sweep_paths = io.list_stamped_files(ref.lidar_dir)
    image_ns, image_paths = io.list_stamped_files(ref.camera_dir("aux"))
    capture_ns = image_ns - int(round(calib.CAMERA_DELAY_S * 1e9))
    match = match_images_to_sweeps(sweep_ns, capture_ns)
    in_range = np.zeros(len(sweep_ns), dtype=bool)
    if len(match):
        t_img = np.where(match >= 0, capture_ns[np.maximum(match, 0)], 0) / 1e9
        in_range = (
            (match >= 0)
            & (t_img - _SLICE_HALF_WIDTH_S >= ego.t[0])
            & (t_img + _SLICE_HALF_WIDTH_S <= ego.t[-1])
        )
    logging.info(
        "MUHOS set %s: %d sweeps, %d colour images; %d frames (%d sweeps without an "
        "image during them, %d outside the vehicle-bus log).",
        ref.set_id,
        len(sweep_ns),
        len(image_ns),
        int(in_range.sum()),
        int(np.sum(match < 0)),
        int(np.sum((match >= 0) & ~in_range)),
    )
    frame_index = 0
    for k in np.flatnonzero(in_range):
        stamp = int(image_ns[match[k]])
        t = int(capture_ns[match[k]]) / 1e9
        x, y, heading, speed = ego.at(np.array([t]))
        window = (ego.t >= t - _SLICE_HALF_WIDTH_S) & (ego.t <= t + _SLICE_HALF_WIDTH_S)
        yield FrameRef(
            set_id=ref.set_id,
            frame_index=frame_index,
            lidar_path=sweep_paths[k],
            lidar_stamp_ns=int(sweep_ns[k]),
            image_path=image_paths[match[k]],
            image_stamp_ns=stamp,
            timestamp=t,
            pose=(float(x[0]), float(y[0]), float(heading[0]), float(speed[0])),
            trajectory_slice=np.stack(
                [ego.t[window], ego.x[window], ego.y[window], ego.heading[window]],
                axis=1,
            ),
            anchor_lat=ego.anchor_lat,
            anchor_lon=ego.anchor_lon,
        )
        frame_index += 1


class MuhosDatasetConverter(SourceDatasetConverter):
    """Iterates the MUHOS driving sets frame by frame."""

    @property
    def multiprocessing_start_method(self) -> str:
        # The worker hot path is laspy/lazrs + cv2 + numpy; nothing TF-like runs in the
        # parent, so ``fork`` is safe and avoids the spawn import cost.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[FrameRef]:
        sets = io.discover_sets(self._input_path)
        if not sets:
            raise FileNotFoundError(
                "No MUHOS driving set (a folder with lidar_points/ and can/) under "
                f"{self._input_path}."
            )
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first set.")
            sets = sets[:1]
        logging.info(
            "Found %d MUHOS set(s): %s", len(sets), ", ".join(s.set_id for s in sets)
        )
        return self._iter_frames(sets)

    @staticmethod
    def _iter_frames(sets: list[io.SetRef]) -> Iterator[FrameRef]:
        for ref in sets:
            if ref.imu_json is None:
                raise FileNotFoundError(
                    f"MUHOS set {ref.path} has no imu/lidar_imu.json: the lidar IMU "
                    "is needed to align the CAN/GPS logs to the sensor clock (they are "
                    "0.3-1.3 s off); extract the set's imu.tar."
                )
            ego = build_ego_trajectory(
                io.read_json(ref.can_json),
                io.read_json(ref.gps_json),
                io.read_imu(ref.imu_json),
            )
            logging.info(
                "MUHOS set %s: motion from %s; bus log %+.3f s off the sensor clock "
                "(drift %+.0f ppm, %d windows); speed scale %.4f, yaw-rate bias "
                "%.3f deg/s, |trajectory - GPS| median %.2f m, p95 %.2f m.",
                ref.set_id,
                ego.source,
                ego.clock.lag0,
                ego.clock.drift * 1e6,
                ego.clock.n_windows,
                ego.speed_scale,
                np.degrees(ego.yaw_rate_bias),
                float(np.median(ego.gps_residual_m)),
                float(np.percentile(ego.gps_residual_m, 95)),
            )
            yield from iter_set_frames(ref, ego)
