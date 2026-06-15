"""Source-dataset converter for View-of-Delft.

Resolves the extracted ``lidar/`` tree (see :mod:`._vod_io`; the detection zip
must be unzipped first) and yields one :class:`._vod_io.FrameRef` per present
keyframe of the requested split, ordered scene-major / frame-ascending. The
iterator is a lazy generator so the full frame list is never materialised; with
``STANDARD_E2E_DEBUG=true`` only the first scene of the split is processed.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.vod._vod_io import (
    FrameRef,
    frame_refs_for_scenes,
    resolve_root,
)
from standard_e2e.caching.src_datasets.vod._vod_splits import scenes_for_split


class VodDatasetConverter(SourceDatasetConverter):
    """Iterates extracted VoD scenes frame-by-frame for one split."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + numpy + json only; no
        # TensorFlow ops fire in the worker, so ``fork`` is safe and avoids the
        # spawn import tax.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[FrameRef]:
        root = resolve_root(self._input_path)
        scenes = scenes_for_split(self._split)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scene.")
            scenes = scenes[:1]
        logging.info(
            "VoD: %d scene(s) for split %r under %s.", len(scenes), self._split, root
        )
        return frame_refs_for_scenes(root, scenes)
