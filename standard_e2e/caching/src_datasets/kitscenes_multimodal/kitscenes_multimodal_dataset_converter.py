"""Source-dataset converter for KITScenes Multimodal.

Resolves the scene directories for the requested split (see :mod:`._kitscenes_io`;
the per-split tarballs must be extracted first) and yields one
``(SceneRef, frame_index)`` per synchronized frame, ordered scene-major /
frame-ascending. The iterator is a lazy generator so the full frame list is never
materialised; with ``STANDARD_E2E_DEBUG=true`` only the first scene of the split
is processed.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.kitscenes_multimodal import _kitscenes_io as io


class KITScenesMultimodalDatasetConverter(SourceDatasetConverter):
    """Iterates extracted KITScenes scenes frame-by-frame for one split."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + pyarrow (parquet) + pyproj +
        # numpy only; no TensorFlow ops fire in the worker, so ``fork`` is safe
        # and avoids the spawn import tax.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[tuple[io.SceneRef, int]]:
        scene_dirs = io.resolve_scene_dirs(self._input_path, self._split)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scene.")
            scene_dirs = scene_dirs[:1]
        logging.info(
            "KITScenes: %d scene(s) for split %r under %s.",
            len(scene_dirs),
            self._split,
            self._input_path,
        )
        return io.frames_for_scene_dirs(scene_dirs)
