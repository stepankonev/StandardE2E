import os

from standard_e2e.caching import TFRecSourceDatasetConverter


class WaymoPerceptionDatasetConverter(TFRecSourceDatasetConverter):
    """Converter for the Waymo Perception dataset."""

    def _get_processing_files(self):
        """Return a list of files to process."""
        return [os.path.join(self._input_path, self._split, "*.tfrecord")]
