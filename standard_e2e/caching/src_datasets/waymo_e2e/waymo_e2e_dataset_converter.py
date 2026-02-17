import logging
import os

from standard_e2e.caching import TFRecSourceDatasetConverter


class WaymoE2EDatasetConverter(TFRecSourceDatasetConverter):
    """Converter for the Waymo E2E dataset."""

    def __init__(
        self,
        source_processor,
        input_path,
        split,
        num_workers=None,
        do_parallel_processing=True,
        arguments=None,
    ):
        super().__init__(
            source_processor,
            input_path,
            split,
            num_workers,
            do_parallel_processing,
            arguments=arguments,
        )
        if self._args.n_shards > 1:
            logging.warning(
                """
                Using sharding with Waymo E2E dataset.
                The frames in the Waymo E2E dataset are stored in a shuffled order.
                When using sharded data, consecutive frames are not guaranteed
                to be contiguous, which can negatively affect
                use cases that require sequential frames.
                For such scenarios, consider preparing and using the full,
                unsharded dataset.
                """
            )

    def _get_processing_files(self):
        """Return a list of files to process."""
        return [os.path.join(self._input_path, f"{self._split}*.tfrecord*")]
