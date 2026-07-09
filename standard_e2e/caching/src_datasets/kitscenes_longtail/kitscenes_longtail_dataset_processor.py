"""Source-dataset processor for KITScenes LongTail.

KITScenes LongTail (KIT / MRT, arXiv 2603.23607) is a sibling of KITScenes
Multimodal focused on **long-tail driving scenarios** (adverse weather,
construction, night, overtaking, ...). Unlike Multimodal's per-scene sensor
directories, LongTail ships as Hugging Face ``datasets`` parquet, one ~9 s
**scenario per row**: six ring-camera image *sequences* (a 360deg multi-view
video), a high-level ``driving_instruction``, an expert future trajectory,
several **counterfactual** trajectories, and multilingual reasoning traces.

Each scenario is **unrolled into one frame per timestep** (segment = scenario),
so the multi-view video is preserved as a real frame sequence. Mapping to
``StandardFrameData`` for frame ``i`` (``i = 0 .. n-1``, the last frame ``n-1``
being the prediction time ``t=0``):

* **cameras**: the six ``camera_*`` ring views at timestep ``i``, keyed by the
  canonical :class:`CameraDirection`; fixed-rig pinhole ``K`` /
  ``T_ego_from_camera`` from :mod:`._kitscenes_longtail_calib`.
* **past_states**: the ego history up to ``i``, re-expressed **ego-relative to
  frame ``i``** (ego at the origin, facing +x along its path tangent), so each
  frame is properly ego-centric; metres, 5 Hz.
* **intent**: the ``driving_instruction`` (scenario-level command) folded into
  :class:`Intent`, on every frame.
* **future_states** / **preference_trajectory**: the 5 s expert future and the
  present counterfactuals (``wrong_speed`` / ``neglect_instruction`` /
  ``off_road`` / ``crash``) -- the prediction targets, attached to the **t=0
  (last) frame only** (and ``future_states`` is withheld for the whole ``test``
  split). The raw instruction, ``scenario_type`` and a prediction-frame flag are
  kept in ``aux_data`` / ``extra_index_data``.

Ego frame: FLU (x-forward, y-left), the frame the trajectories already live in.
Not ingested: lidar / HD map / 3D boxes (LongTail ships none); the Spanish /
Chinese reasoning traces (only English is surfaced into ``aux_data``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from standard_e2e.caching import SourceDatasetProcessor
from standard_e2e.caching.adapters import (
    AbstractAdapter,
    CamerasIdentityAdapter,
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
)
from standard_e2e.caching.src_datasets.kitscenes_longtail import (
    _kitscenes_longtail_io as io,
)
from standard_e2e.caching.src_datasets.kitscenes_longtail._kitscenes_longtail_calib import (  # noqa: E501
    LongTailCameraCalibration,
    build_calibrations,
)
from standard_e2e.constants import PREFERENCE_TRAJECTORIES_KEY
from standard_e2e.data_structures import CameraData, StandardFrameData, Trajectory
from standard_e2e.enums import CameraDirection, Intent, StandardFrameDataField
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import IndexDataGenerator

# 5 Hz trajectory + frame sampling (past = 4 s / 21 pts, futures = 5 s / 25 pts).
_DELTA_T = 0.2

# Non-image columns always read; image columns are added only when cameras are
# consumed (decoding the six sequences is the expensive part).
_BASE_COLUMNS = ["scenario_id", "driving_instruction", "scenario_type", "trajectory"]
_PREFERENCE_LABELS_KEY = "preference_trajectory_labels"


def intent_from_instruction(instruction: str) -> Intent:
    """Fold a free-form ``driving_instruction`` into the coarse :class:`Intent`.

    The raw instruction is the faithful signal (kept in ``aux_data``); this is a
    lossy convenience for the ``intent`` modality: ``left`` -> ``GO_LEFT``,
    ``right`` -> ``GO_RIGHT``, ``straight`` -> ``GO_STRAIGHT``, else ``UNKNOWN``.
    """
    low = instruction.lower()
    if "left" in low:
        return Intent.GO_LEFT
    if "right" in low:
        return Intent.GO_RIGHT
    if "straight" in low:
        return Intent.GO_STRAIGHT
    return Intent.UNKNOWN


def ego_relative_history(past_xy: np.ndarray, frame_index: int) -> Trajectory:
    """Ego history up to ``frame_index``, expressed ego-relative to that frame.

    ``past_xy`` is the scenario's ``(N, 2)`` past in the ``t=0`` ego frame. For an
    earlier frame the ego sits at ``past_xy[frame_index]`` heading along its path
    tangent, so the history is rotated/translated into that frame (ego at the
    origin, facing +x). The last frame (``t=0``) is already the canonical ego
    frame -- where the expert future is expressed -- so its history is kept as-is
    for consistency with that future.
    """
    history = past_xy[: frame_index + 1]
    if frame_index >= len(past_xy) - 1:
        rel = history
    else:
        origin = past_xy[frame_index]
        tangent = past_xy[frame_index + 1] - origin
        theta = float(np.arctan2(tangent[1], tangent[0]))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # (q - origin) @ [[cos, -sin], [sin, cos]] == R(-theta) (q - origin).
        rel = (history - origin) @ np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    timestamps = _DELTA_T * (np.arange(len(history)) - (len(history) - 1))
    return Trajectory(data={TC.X: rel[:, 0], TC.Y: rel[:, 1], TC.TIMESTAMP: timestamps})


class KITScenesLongTailDatasetProcessor(SourceDatasetProcessor):
    """Processor for the KITScenes LongTail dataset.

    The fixed-rig calibration is built once; the current scenario's parquet row is
    cached so its unrolled frames re-use a single read, and shard handles are
    cached per worker.
    """

    DATASET_NAME = "kitscenes_longtail"

    def __init__(
        self,
        common_output_path: str,
        split: str,
        index_data_generator: IndexDataGenerator | None = None,
        adapters: list[AbstractAdapter] | None = None,
        context_aggregators=None,
    ):
        super().__init__(
            common_output_path=common_output_path,
            split=split,
            index_data_generator=index_data_generator,
            adapters=adapters,
            context_aggregators=context_aggregators,
        )
        self._calibs: dict[CameraDirection, LongTailCameraCalibration] = (
            build_calibrations()
        )
        # Per-worker caches (populated lazily in the worker).
        self._parquet_cache: dict[str, Any] = {}
        self._cached_key: tuple[str, int] | None = None
        self._cached_row: dict[str, Any] = {}

    @property
    def dataset_name(self) -> str:
        return self.DATASET_NAME

    @property
    def allowed_splits(self) -> list[str]:
        # Released so far: test / test_raw + 3 train / train_raw samples ("raw" =
        # native-resolution frames; the calibration matches the processed, non-raw
        # frames). val / full train ship in a later dataset version.
        return ["train", "test", "train_raw", "test_raw", "val"]

    def _get_default_adapters(self) -> list[AbstractAdapter]:
        return [
            CamerasIdentityAdapter(),
            IntentIdentityAdapter(),
            PastStatesIdentityAdapter(),
            FutureStatesIdentityAdapter(),
            PreferenceTrajectoryAdapter(),
        ]

    # --- helpers -----------------------------------------------------------

    def _parquet(self, shard_path: str):
        handle = self._parquet_cache.get(shard_path)
        if handle is None:
            import pyarrow.parquet as pq

            handle = pq.ParquetFile(shard_path)
            self._parquet_cache[shard_path] = handle
        return handle

    def _columns(self) -> list[str]:
        columns = list(_BASE_COLUMNS)
        if self.needs_attr(StandardFrameDataField.CAMERAS):
            columns += [
                f"{io.CAMERA_COLUMN_PREFIX}{c.camera_key}"
                for c in self._calibs.values()
            ]
        return columns

    def _row_for(self, ref: io.FrameRef) -> dict[str, Any]:
        """Return the scenario row for ``ref``, caching it so the scenario's
        unrolled frames share one parquet read."""
        key = (ref.shard_path, ref.row_group)
        if key != self._cached_key:
            self._cached_row = io.read_scenario_row(
                self._parquet(ref.shard_path), ref.row_group, columns=self._columns()
            )
            self._cached_key = key
        return self._cached_row

    def _build_cameras(
        self, row: dict, frame_index: int
    ) -> dict[CameraDirection, CameraData]:
        cameras: dict[CameraDirection, CameraData] = {}
        for direction, calib in self._calibs.items():
            cell = row[f"{io.CAMERA_COLUMN_PREFIX}{calib.camera_key}"]
            image = io.decode_frame(cell, frame_index)
            if image is None:
                continue
            cameras[direction] = CameraData(
                camera_direction=direction,
                image=image,
                intrinsics=calib.intrinsics,
                extrinsics=calib.extrinsics.astype("float32"),
                is_fisheye=False,
            )
        return cameras

    def _future(self, traj: dict, key: str) -> Optional[Trajectory]:
        xy, valid = io.trajectory_xy(traj, key)
        if not valid or len(xy) == 0:
            return None
        timestamps = _DELTA_T * np.arange(1, len(xy) + 1)
        return Trajectory(
            data={TC.X: xy[:, 0], TC.Y: xy[:, 1], TC.TIMESTAMP: timestamps}
        )

    def _preference(self, traj: dict) -> tuple[Optional[list[Trajectory]], list[str]]:
        preference: list[Trajectory] = []
        labels: list[str] = []
        for key in io.COUNTERFACTUAL_KEYS:
            trajectory = self._future(traj, key)
            if trajectory is not None:
                preference.append(trajectory)
                labels.append(key)
        return (preference or None), labels

    # --- main entry point --------------------------------------------------

    def _prepare_standardized_frame_data(
        self, raw_frame_data: Any
    ) -> StandardFrameData:
        ref: io.FrameRef = raw_frame_data
        row = self._row_for(ref)
        traj = row["trajectory"]
        instruction = row["driving_instruction"]
        n_frames = len(traj[io.PAST_KEY])
        is_prediction_frame = ref.frame_index >= n_frames - 1

        cameras = (
            self._build_cameras(row, ref.frame_index)
            if self.needs_attr(StandardFrameDataField.CAMERAS)
            else {}
        )

        past_states = None
        if self.needs_attr(StandardFrameDataField.PAST_STATES):
            past_xy, _ = io.trajectory_xy(traj, io.PAST_KEY)
            past_states = ego_relative_history(past_xy, ref.frame_index)

        # Prediction targets live on the t=0 (last) frame only.
        future_states: Optional[Trajectory] = None
        preference: Optional[list[Trajectory]] = None
        labels: list[str] = []
        if is_prediction_frame:
            if self.needs_attr(StandardFrameDataField.FUTURE_STATES):
                future_states = self._future(traj, io.EXPERT_KEY)
            preference, labels = self._preference(traj)

        intent = (
            intent_from_instruction(instruction)
            if self.needs_attr(StandardFrameDataField.INTENT)
            else None
        )
        return StandardFrameData(
            dataset_name=self.dataset_name,
            segment_id=str(row["scenario_id"]),
            frame_id=ref.frame_index,
            timestamp=ref.frame_index * _DELTA_T,
            split=self._split,
            cameras=cameras,
            intent=intent,
            past_states=past_states,
            future_states=future_states,
            aux_data={
                PREFERENCE_TRAJECTORIES_KEY: preference,
                _PREFERENCE_LABELS_KEY: labels,
                "driving_instruction": instruction,
                "scenario_type": row["scenario_type"],
            },
            extra_index_data={
                "scenario_type": str(row["scenario_type"]),
                "driving_instruction": str(instruction),
                "is_prediction_frame": is_prediction_frame,
                f"has_{PREFERENCE_TRAJECTORIES_KEY}": preference is not None,
            },
        )
