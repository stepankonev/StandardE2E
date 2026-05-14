import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tqdm import tqdm

from standard_e2e.caching import TFRecSourceDatasetConverter

# pylint: disable=no-name-in-module
from standard_e2e.third_party.waymo_open_dataset.dataset_pb2 import Frame as WaymoFrame


class WaymoPerceptionDatasetConverter(TFRecSourceDatasetConverter):
    """Converter for the Waymo Perception dataset.

    Pre-populates the processor's per-segment HD-map cache at ``__init__``
    time. Waymo's proto puts ``map_features`` only on the first frame of
    each segment, and per-instance state on the processor is pickled
    independently to every multiprocessing worker -- so without the
    pre-scan, workers that don't happen to receive frame 0 would see an
    empty cache. The pre-scan reads only the first record of each
    tfrecord (~10 ms per file) and runs in the parent process, so the
    cache is part of the pickled state every worker inherits. Cost on
    the training split (~800 segments) is ~8 s.
    """

    # ``forkserver``: a single helper process imports the heavy ML stack
    # (TF, numpy, av2, etc.) once, and each worker is forked from that
    # helper before any TF op fires inside it. That sidesteps the
    # ``fork`` deadlock (parent's already-active TF threads being inherited
    # by workers) while only paying the import cost once instead of once
    # per worker, as ``spawn`` does. Empirically validated on a Waymo
    # Perception 1-tfrecord smoke run: ``fork`` hangs at 0 frames after
    # 180 s; ``forkserver`` completes in 43 s.
    @property
    def multiprocessing_start_method(self) -> str:
        return "forkserver"

    # The prescan cache is now disk-spilled (see ``_prescan_maps``); each
    # worker lazy-loads only the segments it touches rather than
    # receiving the full ~200 MB through ``Pool.initializer``. With the
    # per-worker initial-pickle cost removed, par-32 no longer regresses
    # below par-8 on this dataset, so the previous cap is unnecessary.
    @property
    def max_workers(self) -> Optional[int]:
        return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Path of the transient HD-map prescan cache directory. ``None``
        # when no adapter consumes ``hd_map`` (no prescan needed).
        # Removed in ``_cleanup_after_convert`` once both the frame stage
        # and the aggregator stage finish.
        self._owned_cache_dir: Optional[str] = None
        # The HD-map prescan reads frame 0 of every tfrecord (~3 s/file
        # on HDD; ~40 min on the full training split) to populate the
        # processor's ``_segment_map_cache``. The cache is only ever
        # read inside ``_build_hd_map``, which is itself gated on
        # ``needs_attr("hd_map")``. Skip the prescan entirely when no
        # adapter consumes ``hd_map`` — cameras-only / lidar-only chains
        # save the full prescan cost.
        if self._source_processor.needs_attr("hd_map"):
            # Tell the processor to spill prescan results to disk so the
            # in-memory cache stays empty and ``Pool.initializer`` doesn't
            # ship hundreds of MB to every worker. Workers lazy-load per
            # segment from these files on first hit.
            cache_dir = self._pick_map_cache_dir()
            self._owned_cache_dir = cache_dir
            # Concrete subclass attribute; not exposed on the abstract
            # base.
            setattr(self._source_processor, "_segment_cache_dir", cache_dir)
            logging.info("HD-map prescan cache dir: %s", cache_dir)
            self._prescan_maps()

    def _pick_map_cache_dir(self) -> str:
        """Pick a fast random-access location for the prescan scratch cache.

        Order of preference:

        1. ``/dev/shm`` — Linux ``tmpfs``, RAM-backed, fastest random
           access. Usually half of system RAM (plenty for the typical
           ~200 MB cache); we skip it if free space looks too tight.
        2. ``<output_path>`` — same volume as the produced ``.npz`` files,
           which is assumed to be a fast SSD/NVMe (the same volume that
           workers will be writing to anyway).
        3. ``tempfile.gettempdir()`` — last resort. Often ``/tmp``,
           which may sit on a slow rotational disk; the disk-spill
           pattern relies on fast random-access reads so this is the
           least-preferred choice.

        Returns an exclusive directory created via ``tempfile.mkdtemp``;
        :meth:`_cleanup_after_convert` removes it once ``convert()``
        finishes.
        """
        estimated_bytes = (
            500 * 1024 * 1024
        )  # generous upper bound, full split is ~186 MB
        candidates: list[Path] = [
            Path("/dev/shm"),
            Path(self._source_processor.specific_output_path),
        ]
        for cand in candidates:
            if not cand.is_dir():
                continue
            try:
                free = shutil.disk_usage(cand).free
            except OSError:
                continue
            if free < estimated_bytes:
                logging.info(
                    "Skipping %s for HD-map cache: only %.0f MB free",
                    cand,
                    free / 1e6,
                )
                continue
            return tempfile.mkdtemp(prefix="se2e_wp_map_cache_", dir=str(cand))
        # Last resort — system default tempdir (often /tmp).
        logging.warning(
            "Falling back to system tempdir for HD-map cache; if /tmp is "
            "on a slow disk this will hurt par-32 throughput."
        )
        return tempfile.mkdtemp(prefix="se2e_wp_map_cache_")

    def _cleanup_after_convert(self) -> None:
        """Remove the transient HD-map prescan cache directory, if any."""
        if not self._owned_cache_dir:
            return
        if os.path.isdir(self._owned_cache_dir):
            try:
                shutil.rmtree(self._owned_cache_dir)
                logging.info(
                    "Removed transient HD-map cache dir %s", self._owned_cache_dir
                )
            except OSError as e:
                logging.warning(
                    "Failed to remove HD-map cache dir %s: %s",
                    self._owned_cache_dir,
                    e,
                )
        self._owned_cache_dir = None

    def _get_processing_files(self):
        """Return a list of files to process."""
        return [os.path.join(self._input_path, self._split, "*.tfrecord")]

    def _prescan_maps(self) -> None:
        """Read frame 0 of every tfrecord to populate the processor map cache."""
        cache_method = getattr(
            self._source_processor, "_cache_map_features_for_segment", None
        )
        if cache_method is None:
            return
        files = tf.io.matching_files(self._get_processing_files()).numpy().tolist()
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            files = files[:1]
        logging.info("Pre-scanning %d tfrecord(s) for HD-map features", len(files))

        # Read frame 0 of each tfrecord in a small thread pool. The work is
        # mostly HDD I/O (open + read first record); the kernel's NCQ can
        # serve several reads concurrently, and Python releases the GIL for
        # the blocking syscall. ``ParseFromString`` is CPU-bound and holds
        # the GIL, but the per-file proto is small enough that the I/O
        # parallelism still dominates. The result-writing step
        # (``cache_method``) is serialized below to avoid a race on the
        # processor's ``_segment_map_cache`` dict.
        n_threads = int(os.environ.get("WP_PRESCAN_THREADS", "8"))

        def _read_first(f):
            for raw in tf.data.TFRecordDataset([f]).take(1):
                frame = WaymoFrame()
                frame.ParseFromString(raw.numpy())
                return frame.context.name, list(frame.map_features)
            return None, []

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for seg_name, map_features in tqdm(
                pool.map(_read_first, files),
                total=len(files),
                desc="Pre-scanning HD maps",
            ):
                if seg_name is not None and map_features:
                    cache_method(seg_name, map_features)
