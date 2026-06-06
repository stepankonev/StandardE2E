"""Source-dataset converter for TruckDrive.

Discovers extracted scene directories (see :mod:`._truckdrive_io`; the
per-modality zips must be unzipped first) and yields one ``(SceneRef,
sync_id)`` task per synchronized frame, ordered **scene-major /
sync-ascending**. A frame is every ``sync_id`` that carries an ego pose in
``poses/gt_trajectory.txt``; cameras, LiDAR and boxes are attached per
``sync_id`` by the processor when present.

The frame iterator is a lazy generator so the full task list is never
materialised. With ``STANDARD_E2E_DEBUG=true`` only the first scene is
processed.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.truckdrive._truckdrive_io import (
    SceneRef,
    discover_scenes,
    read_pose_sync_ids,
)


class TruckDriveDatasetConverter(SourceDatasetConverter):
    """Iterates extracted TruckDrive scenes frame-by-frame."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + numpy + json only; no
        # TensorFlow ops fire in the worker, so ``fork`` is safe and avoids the
        # spawn import tax.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[tuple[SceneRef, int]]:
        scenes = discover_scenes(self._input_path)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scene.")
            scenes = scenes[:1]
        logging.info("Found %d TruckDrive scene(s).", len(scenes))
        return self._iter_frames(scenes)

    @staticmethod
    def _iter_frames(scenes: list[SceneRef]) -> Iterator[tuple[SceneRef, int]]:
        for ref in scenes:
            for sync_id in read_pose_sync_ids(ref):
                yield (ref, sync_id)
