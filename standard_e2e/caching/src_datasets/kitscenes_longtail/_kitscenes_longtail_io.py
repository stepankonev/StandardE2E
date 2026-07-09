"""On-disk access helpers for KITScenes LongTail (Hugging Face parquet).

LongTail ships as Hugging Face ``datasets`` parquet shards
(``data/<split>-*.parquet``), one **scenario per row** (and one row-group per
row, so a scenario can be streamed without loading a whole multi-GB shard). Each
scenario row carries:

* ``scenario_id`` (str), ``driving_instruction`` (str), ``scenario_type`` (str);
* six ring-camera image sequences ``frames_camera_<key>`` -- each a list of
  ``{"bytes": <jpeg>, "path": ...}`` at 5 Hz over the 4 s observation window; the
  **last** frame is the prediction time ``t=0``;
* a ``trajectory`` struct: ``past`` (4 s history, 21 pts @ 5 Hz, aligned 1:1 with
  the camera frames) and the futures ``expert_like`` + counterfactuals
  ``wrong_speed`` / ``neglect_instruction`` / ``off_road`` / ``crash`` (5 s, 25
  pts), each ``(N, 2)`` ego-relative metres (x forward, y left). A ``(1, 2)``
  ``[[-100, -100]]`` array is a **withheld / not-applicable** sentinel (all
  futures are withheld in the ``test`` split);
* multilingual ``reasoning`` traces (english / spanish / chinese).

Each scenario is a short multi-view video, so it is **unrolled into one frame
per timestep**: this module enumerates scenarios as tiny ``FrameRef`` (shard path
+ row-group + frame index); the processor caches the current scenario's row so a
scenario's frames re-use one parquet read. Row -> ``StandardFrameData`` math
lives in the processor.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

from standard_e2e.utils import decode_image_bytes

# Per-camera image columns and the trajectory sub-keys.
CAMERA_COLUMN_PREFIX = "frames_camera_"
PAST_KEY = "past"
EXPERT_KEY = "expert_like"
COUNTERFACTUAL_KEYS = ("wrong_speed", "neglect_instruction", "off_road", "crash")
# A (1, 2) [[-100, -100]] trajectory marks a withheld / not-applicable future.
_SENTINEL_VALUE = -100.0


@dataclass(frozen=True)
class FrameRef:
    """Picklable locator for one LongTail timestep (a frame of one scenario)."""

    shard_path: str
    row_group: int
    frame_index: int


def _data_dir(input_path: str) -> str:
    """Directory holding the ``<split>-*.parquet`` shards (``<root>/data`` or the
    path itself)."""
    nested = os.path.join(input_path, "data")
    return nested if os.path.isdir(nested) else input_path


def shards_for_split(input_path: str, split: str) -> list[str]:
    """Sorted parquet shards for ``split`` (matches ``<split>-*.parquet``)."""
    data_dir = _data_dir(input_path)
    return sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))


def scenario_frame_count(parquet_file: pq.ParquetFile, row_group: int) -> int:
    """Number of timesteps in a scenario = its ``past`` length (== camera
    sequence length), read cheaply from the small ``trajectory`` column alone."""
    traj = parquet_file.read_row_group(row_group, columns=["trajectory"]).to_pylist()[0]
    return len(traj["trajectory"][PAST_KEY])


def iter_frame_refs(input_path: str, split: str) -> Iterator[FrameRef]:
    """Yield one ``FrameRef`` per timestep, scenario- then frame-major.

    Raises ``FileNotFoundError`` -- pointing at download -- when no shard matches.
    """
    shards = shards_for_split(input_path, split)
    if not shards:
        raise FileNotFoundError(
            f"No KITScenes-LongTail '{split}-*.parquet' shards under "
            f"{_data_dir(input_path)}. Download the split first "
            f"(see scripts/prepare_dataset_kitscenes_longtail.sh)."
        )
    for shard_path in shards:
        parquet_file = pq.ParquetFile(shard_path)
        for row_group in range(parquet_file.num_row_groups):
            n_frames = scenario_frame_count(parquet_file, row_group)
            for frame_index in range(n_frames):
                yield FrameRef(shard_path, row_group, frame_index)


def read_scenario_row(
    parquet_file: pq.ParquetFile, row_group: int, columns: Optional[list[str]] = None
) -> dict[str, Any]:
    """Read one scenario row (a row-group) into a plain dict."""
    row: dict[str, Any] = parquet_file.read_row_group(
        row_group, columns=columns
    ).to_pylist()[0]
    return row


def decode_frame(camera_cell: list, frame_index: int) -> Optional[np.ndarray]:
    """Decode timestep ``frame_index`` of a camera sequence to an RGB array.

    The index is clamped to the last available frame (the per-camera sequence
    length can differ by one -- e.g. an appended duplicate "now" frame).
    Returns None for an empty sequence.
    """
    if not camera_cell:
        return None
    index = min(frame_index, len(camera_cell) - 1)
    return decode_image_bytes(camera_cell[index]["bytes"])


def trajectory_xy(trajectory: dict[str, Any], key: str) -> tuple[np.ndarray, bool]:
    """Return ``((N, 2) float64 xy, is_valid)`` for a trajectory sub-array.

    ``is_valid`` is False for the ``[[-100, -100]]`` withheld / N-A sentinel.
    """
    arr = np.asarray(trajectory[key], dtype=np.float64).reshape(-1, 2)
    is_valid = not (arr.shape == (1, 2) and np.allclose(arr, _SENTINEL_VALUE))
    return arr, is_valid
