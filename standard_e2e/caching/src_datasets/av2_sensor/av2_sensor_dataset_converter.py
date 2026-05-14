"""Source-dataset converter for the Argoverse 2 (AV2) sensor dataset."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter


class Av2SensorDatasetConverter(SourceDatasetConverter):
    """Iterates AV2 sensor logs frame-by-frame.

    The AV2 sensor split is a flat directory of per-log folders under
    ``<input_path>/<split>/<log_id>/``. One frame = one lidar sweep
    timestamp. Yielding ``(log_dir, sweep_ts_ns)`` ordered first by log
    and then by timestamp keeps consecutive frames inside the same log,
    which lets the processor's per-log cache stay warm across each
    multiprocessing chunk.

    With ``STANDARD_E2E_DEBUG=true`` only the first log is processed,
    matching the convention used by the Waymo converters.
    """

    @property
    def multiprocessing_start_method(self) -> str:
        # AV2 Sensor's worker hot path uses ``cv2`` and pyarrow / numpy only;
        # no TensorFlow ops fire inside the worker, so ``fork`` is safe and
        # avoids the ~5 s/worker import tax incurred by ``spawn``.
        return "fork"

    def _get_source_dataset_iterator(self) -> Iterator[tuple[Path, int]]:
        split_root = Path(self._input_path) / self._split
        if not split_root.is_dir():
            raise FileNotFoundError(f"AV2 sensor split not found at {split_root}")
        log_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
        if not log_dirs:
            raise FileNotFoundError(f"No log directories found under {split_root}")
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info(
                "STANDARD_E2E_DEBUG is set to true, processing only the first log."
            )
            log_dirs = log_dirs[:1]
        logging.info("Found %d AV2 log(s) for split '%s'.", len(log_dirs), self._split)

        # Pre-list all (log_dir, ts) tuples up front rather than globbing
        # each log's lidar dir lazily during iteration. Consolidates HDD
        # directory scans into a bulk pass at converter-init time so they
        # don't interleave with worker reads during processing.
        items: list[tuple[Path, int]] = []
        for log_dir in log_dirs:
            lidar_dir = log_dir / "sensors" / "lidar"
            if not lidar_dir.is_dir():
                logging.warning("No lidar dir for %s; skipping log", log_dir.name)
                continue
            for ts in sorted(int(p.stem) for p in lidar_dir.glob("*.feather")):
                items.append((log_dir, ts))
        logging.info("Pre-listed %d AV2 sensor sweeps.", len(items))
        return iter(items)
