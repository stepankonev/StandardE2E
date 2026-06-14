"""Source-dataset converter for nuScenes (v1.0).

Expected on-disk layout under ``input_path`` (the standard nuScenes dataroot)::

    <version>/*.json          metadata tables (e.g. v1.0-mini/, v1.0-trainval/)
    samples/<CHANNEL>/*.jpg    keyframe camera images
    samples/LIDAR_TOP/*.pcd.bin
    sweeps/                    intermediate frames (unused)

``--split`` is one of the official nuScenes labels (``mini_train`` / ``mini_val``
/ ``train`` / ``val`` / ``test``); it selects both the scene set and the metadata
version (mini_* -> v1.0-mini, train|val -> v1.0-trainval, test -> v1.0-test). The
JSON tables are loaded once here in the parent process and each ``sample``
(keyframe) is resolved into a self-contained :class:`._nuscenes_io.NuscFrame`, so
the workers receive no table state -- the (large, for trainval) metadata is held
once, not per worker.

With ``STANDARD_E2E_DEBUG=true`` only the first scene of the split is processed.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.nuscenes._nuscenes_io import (
    NuscFrame,
    NuscTables,
    iter_resolved_frames,
)
from standard_e2e.caching.src_datasets.nuscenes._nuscenes_splits import (
    VERSION_FOR_SPLIT,
    scenes_for_split,
)


class NuscenesDatasetConverter(SourceDatasetConverter):
    """Iterates nuScenes keyframes (samples) frame-by-frame."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + numpy only; no TensorFlow ops
        # fire in the worker, so ``fork`` is safe and skips the spawn import tax.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[NuscFrame]:
        version = VERSION_FOR_SPLIT.get(self._split)
        if version is None:
            raise ValueError(
                f"Unknown nuScenes split {self._split!r}; expected one of "
                f"{sorted(VERSION_FOR_SPLIT)}."
            )
        tables = NuscTables(self._input_path, version)
        wanted = scenes_for_split(self._split)
        # Scenes of this split present in the metadata, keeping table order (the
        # mini download holds only the mini_* scenes, etc.).
        in_meta = [scene for scene in tables.scenes if scene["name"] in wanted]
        if not in_meta:
            raise FileNotFoundError(
                f"No '{self._split}' scenes found in the {version} metadata at "
                f"{self._input_path}. The mini download holds only the "
                f"mini_train / mini_val scenes."
            )
        # Keep only scenes whose sensor blob is actually on disk -- the trainval
        # *_blobs.tgz arrive one at a time, so the metadata lists all 700 scenes
        # long before their samples/ files exist. This converts a partial
        # download cleanly instead of crashing on the first missing file.
        downloaded = [s for s in in_meta if tables.scene_has_sensor_data(s)]
        if not downloaded:
            raise FileNotFoundError(
                f"The {version} metadata lists {len(in_meta)} '{self._split}' "
                f"scene(s), but none have sensor data under {self._input_path}/"
                f"samples/. Download/extract the *_blobs.tgz for this split."
            )
        if len(downloaded) < len(in_meta):
            logging.info(
                "nuScenes: %d/%d '%s' scenes have sensor data on disk (partial "
                "download) -- converting those.",
                len(downloaded),
                len(in_meta),
                self._split,
            )
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scene.")
            downloaded = downloaded[:1]
        logging.info(
            "Found %d nuScenes scene(s) for split '%s' (%s).",
            len(downloaded),
            self._split,
            version,
        )
        return iter_resolved_frames(tables, {s["name"] for s in downloaded})
