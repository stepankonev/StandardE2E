"""Source-dataset converter for comma2k19.

Discovers segments (extracted directories or ``Chunk_*.zip`` archives; see
:mod:`._comma_io`) and yields one ``(SegmentRef, frame_index)`` task per video
frame, ordered **segment-major / frame-ascending**. That ordering is what lets
the processor decode each segment's HEVC stream forward-only: a worker pulling
tasks off the shared pool queue walks a monotonically increasing subsequence of
a segment's frames.

``--frame_stride`` subsamples the native 20 Hz stream (``1`` keeps every
frame). With ``STANDARD_E2E_DEBUG=true`` only the first segment is processed.

The frame iterator is a lazy generator so the full per-frame task list (~1.2M
frames per chunk) is never materialised in memory.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections import defaultdict
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.comma2k19._comma_io import (
    SegmentRef,
    discover_segments,
    read_frame_times,
)


class Comma2k19DatasetConverter(SourceDatasetConverter):
    """Iterates comma2k19 segments frame-by-frame."""

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
        # Group by source archive/dir so each zip is opened once for the cheap
        # frame-count probe (reading only ``frame_times``).
        by_container: dict[tuple[str, str], list[SegmentRef]] = defaultdict(list)
        for ref in segments:
            by_container[(ref.kind, ref.container)].append(ref)
        for (kind, container), refs in by_container.items():
            zip_handle = zipfile.ZipFile(container) if kind == "zip" else None
            try:
                for ref in refs:
                    n_frames = len(read_frame_times(ref, zip_handle=zip_handle))
                    for frame_idx in range(0, n_frames, stride):
                        yield (ref, frame_idx)
            finally:
                if zip_handle is not None:
                    zip_handle.close()

    def _cleanup_after_convert(self) -> None:
        # Remove the scratch tree of HEVC streams extracted from zip archives.
        scratch_root = getattr(self._source_processor, "_scratch_root", None)
        if scratch_root and os.path.isdir(scratch_root):
            shutil.rmtree(scratch_root, ignore_errors=True)
