"""Source-dataset converter for NAVSIM (OpenScene-v1.1 trainval logs)."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter


class NavsimDatasetConverter(SourceDatasetConverter):
    """Iterates NAVSIM scenes frame-by-frame.

    Expected on-disk layout under ``input_path``::

        navsim_logs/<split>/*.pkl       per-log scene pickles
        sensor_blobs/<split>/<log>/     CAM_F0/ CAM_L{0,1,2}/ CAM_R{0,1,2}/
                                        CAM_B0/ MergedPointCloud/

    Each pickle is a list of frame dicts; one frame = one ego timestamp.
    Yielding ``(log_pickle_path, frame_idx)`` ordered first by log and
    then by frame keeps the processor's per-pickle cache warm across the
    multiprocessing chunk.

    With ``STANDARD_E2E_DEBUG=true`` only the first log is processed.
    """

    def _get_source_dataset_iterator(self) -> Iterator[tuple[Path, int]]:
        log_root = Path(self._input_path) / "navsim_logs" / self._split
        if not log_root.is_dir():
            raise FileNotFoundError(
                f"NAVSIM logs dir not found at {log_root}. Expected layout: "
                f"<input_path>/navsim_logs/<split>/*.pkl"
            )
        log_files = sorted(log_root.glob("*.pkl"))
        if not log_files:
            raise FileNotFoundError(f"No log pickles found under {log_root}")
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info(
                "STANDARD_E2E_DEBUG is set to true, processing only the first log."
            )
            log_files = log_files[:1]
        logging.info(
            "Found %d NAVSIM log(s) for split '%s'.", len(log_files), self._split
        )

        def _iter() -> Iterator[tuple[Path, int]]:
            for log_path in log_files:
                # Length-only peek; the processor caches the full payload itself.
                with open(log_path, "rb") as fp:
                    n_frames = len(pickle.load(fp))
                for frame_idx in range(n_frames):
                    yield (log_path, frame_idx)

        return _iter()
