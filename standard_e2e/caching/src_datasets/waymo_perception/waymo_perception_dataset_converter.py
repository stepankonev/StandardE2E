import logging
import os

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prescan_maps()

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
        for f in tqdm(files, desc="Pre-scanning HD maps"):
            for raw in tf.data.TFRecordDataset([f]).take(1):
                frame = WaymoFrame()
                frame.ParseFromString(raw.numpy())
                if frame.map_features:
                    cache_method(frame.context.name, frame.map_features)
