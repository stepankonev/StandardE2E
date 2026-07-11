"""Source-dataset processor for the NATIX Multi-Camera Driving Dataset.

NATIX Multi-Camera (natix.network) is a crowd-sourced, real-world driving
dataset -- 100 h of Tesla-dashcam surround footage from everyday drivers in
Switzerland and the United States, segmented into ~1-minute mp4 clips at
~36 fps with per-clip GPS metadata (1-10 Hz, consumer-grade), per-trip camera
calibration and trip-level metadata. The StandardE2E dataset key is
``natix_multicam`` (NATIX has announced a telemetry-included sibling release,
which can be added alongside, mirroring the kitscenes_multimodal /
kitscenes_longtail convention). It ships **no** lidar, HD map, 3D boxes or
driving command.

Mapping to ``StandardFrameData``:

* One **frame = one front-camera GPS fix** (~1-10 Hz; video frames between
  fixes carry no pose and are not emitted); one **segment = one trip piece**
  (a ``<trip-id>_<n>[_seqNN]`` directory -- continuous minutes), its clips
  concatenated chronologically (see :mod:`._natix_io`).
* **cameras**: the rig's 4 or 6 views, keyed by the canonical
  :class:`CameraDirection` matching each camera's facing (in 4-camera trips
  ``LEFT``/``RIGHT`` are the backward-facing repeaters -> ``REAR_LEFT`` /
  ``REAR_RIGHT``; pillar cameras face forward-side -> ``FRONT_LEFT`` /
  ``FRONT_RIGHT``). Pinhole ``K`` + Brown-Conrady distortion as shipped
  (the front camera can be natively larger than the trip's standard frame
  size -- its ``K`` already matches the native footage); extrinsics are
  ``T_ego_from_camera`` in the optical frame (see :mod:`._natix_geometry`).
  Because per-camera streams differ in frame rate and duration, each camera
  is matched to the fix by timestamp; a camera with no usable match for a
  frame (missing files, >1 s offset) is simply absent from ``cameras``.
* **global_position** / ``aux_data["pose_matrix"]``: the GPS pose in a
  per-segment local east/north metric frame (azimuthal-equidistant anchored
  at the segment's first fix; z = 0 -- no altitude in the GPS stream), yaw
  from ``heading_deg`` with unreliable headings held through stops. It also
  carries the fix **speed** (``TrajectoryComponent.SPEED``). Past/future ego
  trajectories are produced by :class:`FuturePastStatesFromMatricesAggregator`,
  exactly as for comma2k19 / AV2 / Waymo.
* ``extra_index_data``: the trip name, country and region (from
  ``trip_insight.json``'s first per-minute record -- the directory layout's
  ``Country/[State]`` depth varies and is not parsed) and the rig's camera
  count -- so 4- vs 6-camera trips and geographies are filterable from the
  index.

Data-quality notes (inherited from the dataset, documented rather than
"fixed"): the GPS is consumer-grade (fix accuracy is tens of metres and
video/GPS sync error is 0-1 s, in extremes up to 3 s), so poses are far
noisier than fleet-grade datasets; calibrations are nominal per vehicle
model, not per-vehicle calibrated.

Video decoding: mp4 clips are decoded **forward-only** with per-worker,
per-clip :class:`cv2.VideoCapture` readers (one per camera), the same pattern
as comma2k19 -- the frame iterator is ordered segment-major / frame-ascending,
so each worker walks a monotonically increasing subsequence of every clip.
CSV ``frame_number`` can exceed the decodable frame count by 1-2 frames (mp4
metadata vs stream truth), so a reader that hits end-of-stream returns its
last decoded frame (<= ~60 ms off, far below the dataset's own sync error).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import AbstractAdapter, CamerasIdentityAdapter
from standard_e2e.caching.segment_context import (
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io
from standard_e2e.caching.src_datasets.natix_multicam._natix_geometry import (
    LocalMetricProjection,
    NatixCameraCalibration,
    headings_rad_from_deg,
    parse_fixed_metadata,
    poses_world_from_xy_heading,
    resolve_headings,
)
from standard_e2e.data_structures import CameraData, StandardFrameData, Trajectory
from standard_e2e.enums import CameraDirection, StandardFrameDataField
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator
from standard_e2e.utils import matrix_to_xyz_heading

# Single decode thread per worker process: parallelism is across frames in
# the pool, so per-VideoCapture FFmpeg thread pools only oversubscribe cores.
cv2.setNumThreads(0)


class _ForwardVideoReader:
    """Forward-only mp4 frame reader with a last-frame cache.

    ``read(i)`` decodes ahead to frame ``i`` (0-based) and returns it as RGB.
    Repeats of the current frame return the cached image (timestamp matching
    can assign one video frame to consecutive GPS fixes); a request behind the
    cursor reopens the stream (rare out-of-order task). At end-of-stream the
    last decoded frame is returned instead (CSV frame numbers can overrun the
    stream by 1-2 frames); ``None`` only when nothing was ever decoded.
    """

    def __init__(self, video_path: str):
        self._video_path = video_path
        self._capture: Optional[cv2.VideoCapture] = None
        self._next_index = 0
        self._last_index = -1
        self._last_image: Optional[np.ndarray] = None

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._video_path)
        if not capture.isOpened():
            raise RuntimeError(f"could not open NATIX video {self._video_path}")
        return capture

    def read(self, frame_index: int) -> Optional[np.ndarray]:
        if frame_index == self._last_index and self._last_image is not None:
            return self._last_image
        if self._capture is None or frame_index < self._next_index:
            self.release()
            self._capture = self._open()
            self._next_index = 0
        while self._next_index < frame_index:
            if not self._capture.grab():
                logging.debug(
                    "NATIX: end of stream at frame %d (wanted %d) in %s; "
                    "reusing the last decoded frame.",
                    self._next_index,
                    frame_index,
                    self._video_path,
                )
                return self._last_image
            self._next_index += 1
        ok, bgr = self._capture.read()
        if not ok:
            logging.debug(
                "NATIX: failed to decode frame %d in %s; "
                "reusing the last decoded frame.",
                frame_index,
                self._video_path,
            )
            return self._last_image
        self._next_index = frame_index + 1
        self._last_index = frame_index
        self._last_image = np.ascontiguousarray(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
        return self._last_image

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._next_index = 0


class NatixMulticamDatasetProcessor(SourceDatasetProcessor):
    """Processor for the NATIX Multi-Camera Driving Dataset.

    Per-trip state (calibration, clip list, the aligned frame table and the
    derived pose matrices) is built once and reused across every frame of the
    trip; the cache is keyed by trip name so pooled workers only rebuild on
    transition. Video readers are additionally keyed by the current clip.
    """

    DATASET_NAME = "natix_multicam"

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
        # Per-worker, per-trip cache (populated lazily in the worker).
        self._cached_trip_name: Optional[str] = None
        self._calibrations: dict[CameraDirection, NatixCameraCalibration] = {}
        self._clips: list[io.ClipRef] = []
        self._table: Optional[io.TripFrameTable] = None
        self._poses_world_from_ego = np.zeros((0, 4, 4), dtype=np.float64)
        self._speeds = np.zeros((0,), dtype=np.float64)
        self._country = ""
        self._region = ""
        # Per-clip video readers (forward-only; see the module docstring).
        self._readers: dict[CameraDirection, _ForwardVideoReader] = {}
        self._readers_clip_index: Optional[int] = None

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # NATIX has no canonical train/val/test split; ``split`` is a
        # passthrough output label (as for comma2k19).
        return ["all", "train", "val", "test"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [CamerasIdentityAdapter()]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- per-trip cache -----------------------------------------------------

    def _refresh_trip_cache(self, ref: io.TripRef) -> None:
        if self._cached_trip_name == ref.trip_name:
            return
        self._release_readers()
        self._calibrations = parse_fixed_metadata(io.read_fixed_metadata(ref))
        self._clips = io.discover_clips(ref)
        insight = io.read_trip_insight(ref)
        timezone_name = io.trip_timezone(insight)
        self._country, self._region = io.trip_location(insight)
        table = io.build_trip_frame_table(
            self._clips,
            [calib.folder_key for calib in self._calibrations.values()],
            timezone_name,
        )
        self._table = table

        if table.n_frames:
            projection = LocalMetricProjection(table.latitudes[0], table.longitudes[0])
            x_east, y_north = projection.to_local_xy(table.latitudes, table.longitudes)
            headings_rad = headings_rad_from_deg(
                resolve_headings(table.headings_deg, table.speeds_mps)
            )
            self._poses_world_from_ego = poses_world_from_xy_heading(
                x_east, y_north, headings_rad
            )
        else:
            self._poses_world_from_ego = np.zeros((0, 4, 4), dtype=np.float64)
        # Speed can be missing on a fix even after the velocity fallback;
        # 0 keeps the Trajectory finite (comma2k19 always carries a speed).
        self._speeds = np.nan_to_num(table.speeds_mps, nan=0.0)

        self._cached_trip_name = ref.trip_name
        logging.debug(
            "NATIX trip %s cached: %d frames, %d clips, %d cameras",
            ref.trip_name,
            table.n_frames,
            len(self._clips),
            len(self._calibrations),
        )

    # --- modality builders --------------------------------------------------

    def _reader_for(
        self, direction: CameraDirection, clip_index: int
    ) -> Optional[_ForwardVideoReader]:
        if self._readers_clip_index != clip_index:
            self._release_readers()
            self._readers_clip_index = clip_index
        reader = self._readers.get(direction)
        if reader is None:
            files = self._clips[clip_index].cameras.get(
                self._calibrations[direction].folder_key
            )
            if files is None:
                return None
            reader = _ForwardVideoReader(files.video_path)
            self._readers[direction] = reader
        return reader

    def _build_cameras(self, frame_index: int) -> dict[CameraDirection, CameraData]:
        assert self._table is not None
        clip_index = int(self._table.clip_indices[frame_index])
        cameras: dict[CameraDirection, CameraData] = {}
        for direction, calib in self._calibrations.items():
            frame_number = int(self._table.frame_numbers[calib.folder_key][frame_index])
            # -1 is the no-match sentinel; frame numbers are 1-based, so 0 is
            # equally unusable.
            if frame_number < 1:
                continue
            reader = self._reader_for(direction, clip_index)
            if reader is None:
                continue
            # CSV frame numbers are 1-based and synced to the camera's mp4.
            image = reader.read(frame_number - 1)
            if image is None:
                logging.warning(
                    "NATIX: no decodable frame %d for %s in clip %s; "
                    "omitting the camera.",
                    frame_number,
                    direction,
                    self._clips[clip_index].clip_name,
                )
                continue
            cameras[direction] = CameraData(
                camera_direction=direction,
                image=image,
                intrinsics=calib.intrinsics,
                extrinsics=calib.extrinsics.astype(np.float32),
                distortion=calib.distortion,
                is_fisheye=False,
            )
        return cameras

    # --- main entry point ---------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        ref, frame_index = raw_frame_data
        self._refresh_trip_cache(ref)
        assert self._table is not None
        if not 0 <= frame_index < self._table.n_frames:
            raise IndexError(
                f"frame {frame_index} out of range for trip {ref.trip_name} "
                f"({self._table.n_frames} frames)"
            )

        t_world_ego = self._poses_world_from_ego[frame_index]
        x, y, z, heading = matrix_to_xyz_heading(t_world_ego)
        timestamp = float(self._table.timestamps[frame_index])
        speed = float(self._speeds[frame_index])

        cameras = (
            self._build_cameras(frame_index)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.trip_name,
            frame_id=frame_index,
            timestamp=timestamp,
            split=self._split,
            global_position=Trajectory(
                {
                    TC.TIMESTAMP: [timestamp],
                    TC.X: [x],
                    TC.Y: [y],
                    TC.Z: [z],
                    TC.HEADING: [heading],
                    TC.SPEED: [speed],
                }
            ),
            cameras=cameras,
            aux_data={"pose_matrix": t_world_ego.astype(np.float64)},
            extra_index_data={
                "trip": ref.trip_name,
                "country": self._country,
                "region": self._region,
                "camera_count": len(self._calibrations),
            },
        )

    # --- cleanup ------------------------------------------------------------

    def _release_readers(self) -> None:
        for reader in self._readers.values():
            reader.release()
        self._readers = {}
        self._readers_clip_index = None

    def cleanup(self) -> None:
        """Release the worker-local video handles."""
        self._release_readers()
