"""Source-dataset converter for the NATIX Multi-Camera Driving Dataset.

Discovers trip pieces (no extraction step -- the download is a plain
directory tree; see :mod:`._natix_io`) and yields one ``(TripRef,
frame_index)`` task per emitted frame (one per front-camera GPS fix), ordered
**trip-major / frame-ascending**. That ordering is what lets the processor
decode each clip's mp4 streams forward-only: a worker pulling tasks off the
shared pool queue walks a monotonically increasing subsequence of a trip's
frames and never rewinds.

Frame counts come from the same frame-table build the processor uses (front
CSVs only -- the per-camera matching never changes the row count), so the
indices the converter yields always agree with the processor's table.

``--frame_stride`` subsamples the native ~1-10 Hz fix stream (``1`` keeps
every fix); downscaling, if wanted, is the ``cameras_identity_adapter``'s
``max_size`` param (config-driven). With ``STANDARD_E2E_DEBUG=true`` only the
first trip is processed. The iterator is a lazy generator, so per-trip CSVs
are read as the pipeline reaches them.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.natix_multicam import _natix_io as io


class NatixMulticamDatasetConverter(SourceDatasetConverter):
    """Iterates NATIX trip pieces frame-by-frame."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (mp4 decode) + pandas + pyproj + numpy only;
        # no TF ops fire in the worker, so ``fork`` is safe and avoids the
        # spawn import tax.
        return "fork"

    @classmethod
    def get_arg_parser(cls):
        parser = super().get_arg_parser()
        parser.add_argument(
            "--frame_stride",
            type=int,
            default=1,
            help=(
                "Sample every Nth GPS-fix frame (1 = every fix, ~1-10 Hz). "
                "Use a larger stride to cut output volume / processing time."
            ),
        )
        return parser

    def _get_source_dataset_iterator(self) -> Iterator[tuple[io.TripRef, int]]:
        trips = io.discover_trips(self._input_path)
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first trip.")
            trips = trips[:1]
        stride = max(1, int(getattr(self._args, "frame_stride", 1)))
        logging.info("Found %d NATIX trip(s); frame_stride=%d.", len(trips), stride)
        return self._iter_frames(trips, stride)

    @staticmethod
    def _iter_frames(
        trips: list[io.TripRef], stride: int
    ) -> Iterator[tuple[io.TripRef, int]]:
        for ref in trips:
            clips = io.discover_clips(ref)
            # Front-only build: same row count as the processor's full table
            # (frames are front-fix-anchored), without reading every camera.
            table = io.build_trip_frame_table(
                clips,
                [io.FRONT_CAMERA_KEY],
                io.trip_timezone(io.read_trip_insight(ref)),
            )
            if not table.n_frames:
                logging.warning(
                    "NATIX: trip %s has no usable frames; skipping.", ref.trip_name
                )
                continue
            for frame_index in range(0, table.n_frames, stride):
                yield (ref, frame_index)
