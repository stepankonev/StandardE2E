"""Source-dataset converter for KITScenes LongTail.

Each ~9 s scenario is a short multi-view video, so it is **unrolled into one
frame per timestep**: this yields one ``FrameRef`` (parquet shard + row-group +
frame index) per timestep of the requested split (see
:mod:`._kitscenes_longtail_io`; the parquet shards must be downloaded first). The
iterator is lazy so the multi-GB shards are never materialised; with
``STANDARD_E2E_DEBUG=true`` only the first scenario (all of its frames) is
processed.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.kitscenes_longtail import (
    _kitscenes_longtail_io as io,
)


class KITScenesLongTailDatasetConverter(SourceDatasetConverter):
    """Iterates KITScenes LongTail timesteps (frames of parquet rows) per split."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + pyarrow + numpy only; no
        # TensorFlow ops fire, so ``fork`` is safe and avoids the spawn tax.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[io.FrameRef]:
        refs = io.iter_frame_refs(self._input_path, self._split)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scenario.")
            # All frames of the first scenario (the first row-group).
            first = next(refs, None)
            if first is None:
                return iter(())
            same_scenario = itertools.takewhile(
                lambda r: (r.shard_path, r.row_group)
                == (first.shard_path, first.row_group),
                refs,
            )
            return itertools.chain([first], same_scenario)
        return refs
