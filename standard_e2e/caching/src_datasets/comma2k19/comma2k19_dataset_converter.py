"""Source-dataset converter for comma2k19.

Discovers extracted segment directories (see :mod:`._comma_io`; the
``Chunk_*.zip`` archives must be unzipped first) and yields one
``(SegmentRef, frame_index)`` task per video frame, ordered **segment-major /
frame-ascending**. That ordering is what lets the processor decode each
segment's HEVC stream forward-only: a worker pulling tasks off the shared pool
queue walks a monotonically increasing subsequence of a segment's frames.

``--frame_stride`` subsamples the native 20 Hz stream (``1`` keeps every
frame); ``--image_max_size`` optionally downscales each frame (intrinsics
scaled to match). With ``STANDARD_E2E_DEBUG=true`` only the first segment is
processed.

The frame iterator is a lazy generator so the full per-frame task list (~1.2M
frames per chunk) is never materialised in memory.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.comma2k19._comma_io import (
    SegmentRef,
    discover_segments,
    read_frame_times,
)
from standard_e2e.caching.src_datasets.comma2k19.comma2k19_dataset_processor import (
    Comma2k19DatasetProcessor,
)


class Comma2k19DatasetConverter(SourceDatasetConverter):
    """Iterates extracted comma2k19 segments frame-by-frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Forward the optional downscale knob to the processor before the pool
        # pickles it to the workers.
        max_size = getattr(self._args, "image_max_size", None)
        if max_size is not None and isinstance(
            self._source_processor, Comma2k19DatasetProcessor
        ):
            self._source_processor.image_max_size = int(max_size)

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (HEVC decode) + numpy only; no TF ops fire in
        # the worker, so ``fork`` is safe and avoids the spawn import tax.
        return "fork"

    @classmethod
    def get_arg_parser(cls):
        parser = super().get_arg_parser()
        parser.add_argument(
            "--frame_stride",
            type=int,
            default=1,
            help=(
                "Sample every Nth video frame (1 = every 20 Hz frame). Use a "
                "larger stride to cut output volume / processing time."
            ),
        )
        parser.add_argument(
            "--image_max_size",
            type=int,
            default=None,
            help=(
                "Downscale each frame so its longest side is at most N px "
                "(intrinsics scaled to match). Default keeps native 1164x874."
            ),
        )
        return parser

    def _get_source_dataset_iterator(self) -> Iterator[tuple[SegmentRef, int]]:
        segments = discover_segments(self._input_path)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first segment.")
            segments = segments[:1]
        stride = max(1, int(getattr(self._args, "frame_stride", 1)))
        logging.info(
            "Found %d comma2k19 segment(s); frame_stride=%d.", len(segments), stride
        )
        return self._iter_frames(segments, stride)

    @staticmethod
    def _iter_frames(
        segments: list[SegmentRef], stride: int
    ) -> Iterator[tuple[SegmentRef, int]]:
        for ref in segments:
            n_frames = len(read_frame_times(ref))
            for frame_idx in range(0, n_frames, stride):
                yield (ref, frame_idx)
