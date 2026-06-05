"""Source-dataset processor for comma2k19.

comma2k19 is a large highway-driving dataset: ~2000 one-minute *segments*, each
a single forward-facing 20 Hz road camera (the comma EON, 1164x874) plus a
fused GNSS/IMU ego pose and CAN telemetry. It ships **no** lidar, HD map, 3D
boxes, or driving command. Segments must be extracted from the distributed
``Chunk_*.zip`` archives before processing (see :mod:`._comma_io`).

Mapping to ``StandardFrameData``:

* One frame = one video frame of a segment; one segment = one cache segment.
* **cameras**: the single ``CameraDirection.FRONT`` image at that frame, with
  the EON pinhole intrinsics and identity extrinsics (the dataset's pose *is*
  the camera pose, so the ego frame and the camera coincide).
* **global_position** / ``aux_data["pose_matrix"]``: the ego pose in a
  per-segment local world frame (FLU), derived from the ECEF position +
  orientation in ``global_pose`` (see :mod:`._comma_geometry`). It also carries
  the instantaneous **speed** (``TrajectoryComponent.SPEED``) from the ECEF
  velocity. Past/future ego trajectories are produced by
  :class:`FuturePastStatesFromMatricesAggregator`, exactly as for AV2 / Waymo /
  WayveScenes.

Video decoding: comma2k19 frames live inside per-segment HEVC streams whose
random-access seek is unreliable, so frames are decoded **forward-only** with a
per-worker, per-segment :class:`cv2.VideoCapture` cache. The frame iterator is
ordered segment-major / frame-ascending (see the converter) so each worker
walks a monotonically increasing subsequence of a segment's frames and never
needs to rewind.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Quieten the FFmpeg HEVC decoder OpenCV shells out to (must be set before the
# first VideoCapture). comma2k19's bare .hevc streams otherwise emit per-open
# "missing PPS/SPS"-style notices on stderr.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # noqa: E402  (import after the env var above)
import numpy as np  # noqa: E402

from standard_e2e.caching import SourceDatasetProcessor  # noqa: E402
from standard_e2e.caching.adapters import (  # noqa: E402
    AbstractAdapter,
    CamerasIdentityAdapter,
)
from standard_e2e.caching.segment_context import (  # noqa: E402
    FuturePastStatesFromMatricesAggregator,
    SegmentContextAggregator,
)
from standard_e2e.caching.src_datasets.comma2k19._comma_geometry import (  # noqa: E402
    pose_matrices_world_from_ego,
)
from standard_e2e.caching.src_datasets.comma2k19._comma_io import (  # noqa: E402
    SegmentRef,
    load_pose_arrays,
)
from standard_e2e.data_structures import (  # noqa: E402
    CameraData,
    StandardFrameData,
    Trajectory,
)
from standard_e2e.enums import CameraDirection, StandardFrameDataField  # noqa: E402
from standard_e2e.enums import TrajectoryComponent as TC  # noqa: E402
from standard_e2e.indexing import IndexDataGenerator  # noqa: E402
from standard_e2e.utils import intrinsics_matrix, matrix_to_xyz_heading  # noqa: E402

# Single decode thread per worker process: comma2k19 parallelism is across
# segments/frames in the pool, so letting each VideoCapture spin up its own
# FFmpeg thread pool only oversubscribes the cores.
cv2.setNumThreads(0)


class Comma2k19DatasetProcessor(SourceDatasetProcessor):
    """Processor for the comma2k19 dataset."""

    DATASET_NAME = "comma2k19"
    # comma EON road camera, treated as a pinhole with the dataset's published
    # focal length and a principal point at the image centre.
    CAM_WIDTH = 1164
    CAM_HEIGHT = 874
    EON_FOCAL_LENGTH = 910.0

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
        # Per-worker, per-segment cache (populated lazily in the worker).
        self._cached_segment_id: Optional[str] = None
        self._pose_world_from_ego = np.zeros((0, 4, 4), dtype=np.float64)
        self._speeds = np.zeros((0,), dtype=np.float32)
        self._timestamps = np.zeros((0,), dtype=np.float64)
        self._n_frames = 0
        # Forward-only video reader state.
        self._cap: Optional[cv2.VideoCapture] = None
        self._next_frame_idx = 0
        self._video_path: Optional[str] = None

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # comma2k19 has no canonical train/val/test split; ``split`` is a
        # passthrough output label.
        return ["all", "train", "val", "test"]

    @property
    def camera_intrinsics(self) -> np.ndarray:
        """EON pinhole intrinsics ``K`` (3x3 float32)."""
        focal = self.EON_FOCAL_LENGTH
        return intrinsics_matrix(
            focal, focal, self.CAM_WIDTH / 2.0, self.CAM_HEIGHT / 2.0
        )

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [CamerasIdentityAdapter()]

    def _get_default_context_aggregators(self) -> list[SegmentContextAggregator]:
        return [FuturePastStatesFromMatricesAggregator(self.output_path)]

    # --- per-segment cache ------------------------------------------------

    def _refresh_segment_cache(self, ref: SegmentRef) -> None:
        if self._cached_segment_id == ref.segment_id:
            return
        poses = load_pose_arrays(ref)
        positions = poses["frame_positions"]
        quats = poses["frame_orientations"]
        velocities = poses["frame_velocities"]
        times = np.asarray(poses["frame_times"], dtype=np.float64).reshape(-1)
        # Guard against any rare per-array length mismatch by clipping to the
        # shortest; in practice all global_pose arrays match the frame count.
        n = int(min(len(positions), len(quats), len(velocities), len(times)))

        self._pose_world_from_ego = pose_matrices_world_from_ego(
            positions[:n], quats[:n]
        )
        self._speeds = np.linalg.norm(velocities[:n], axis=1).astype(np.float32)
        self._timestamps = times[:n]
        self._n_frames = n

        self._release_video()
        self._video_path = ref.video_path
        self._cap = None  # opened lazily on first read
        self._next_frame_idx = 0
        self._cached_segment_id = ref.segment_id
        logging.debug("comma2k19 segment %s cached: %d frames", ref.segment_id, n)

    # --- modality builders ------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        if self._video_path is None:
            raise RuntimeError("no video is cached for the current segment")
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open comma2k19 video {self._video_path}")
        return cap

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Decode ``frame_idx`` forward-only and return a contiguous RGB image.

        HEVC keyframe seeking is unreliable here, so we only ever advance: a
        request at or after ``_next_frame_idx`` grabs intervening frames; a
        request before it (a rare out-of-order task) reopens the stream.
        """
        if self._cap is None or frame_idx < self._next_frame_idx:
            if self._cap is not None:
                self._cap.release()
            self._cap = self._open_capture()
            self._next_frame_idx = 0
        while self._next_frame_idx < frame_idx:
            if not self._cap.grab():
                raise RuntimeError(
                    f"unexpected end of stream at frame {self._next_frame_idx} "
                    f"(wanted {frame_idx}) in segment {self._cached_segment_id}"
                )
            self._next_frame_idx += 1
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError(
                f"failed to decode frame {frame_idx} in segment "
                f"{self._cached_segment_id}"
            )
        self._next_frame_idx = frame_idx + 1
        return np.ascontiguousarray(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

    def _build_cameras(self, frame_idx: int) -> dict[CameraDirection, CameraData]:
        return {
            CameraDirection.FRONT: CameraData(
                camera_direction=CameraDirection.FRONT,
                image=self._read_frame(frame_idx),
                intrinsics=self.camera_intrinsics,
                # The pose is the camera pose, so camera == ego (FLU).
                extrinsics=np.eye(4, dtype=np.float32),
                is_fisheye=False,
            )
        }

    def _prepare_standardized_frame_data(self, raw_frame_data) -> StandardFrameData:
        ref, frame_idx = raw_frame_data
        self._refresh_segment_cache(ref)
        if not 0 <= frame_idx < self._n_frames:
            raise IndexError(
                f"frame {frame_idx} out of range for segment "
                f"{ref.segment_id} ({self._n_frames} frames)"
            )

        t_world_ego = self._pose_world_from_ego[frame_idx]
        x, y, z, heading = matrix_to_xyz_heading(t_world_ego)
        timestamp = float(self._timestamps[frame_idx])
        speed = float(self._speeds[frame_idx])

        cameras = (
            self._build_cameras(frame_idx)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )

        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=ref.segment_id,
            frame_id=frame_idx,
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
            extra_index_data={"route": ref.route, "segment": ref.segment},
        )

    # --- cleanup ----------------------------------------------------------

    def _release_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_path = None

    def cleanup(self) -> None:
        """Release the worker-local video handle."""
        self._release_video()
