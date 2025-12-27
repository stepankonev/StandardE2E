from standard_e2e.data_structures import FrameIndexData, TransformedFrameData


class IndexDataGenerator:
    """Class for index data generators."""

    def generate_index_data(self, data: TransformedFrameData) -> FrameIndexData:
        """Build a ``FrameIndexData`` record from a transformed frame.

        Args:
            data: Frame whose metadata should be stored in the index.

        Returns:
            FrameIndexData populated with filename, split, ids, and timestamps.
        """
        if data.filename is None:
            raise ValueError("TransformedFrameData must have a filename.")
        frame_index_data = FrameIndexData(
            dataset_name=data.dataset_name,
            segment_id=data.segment_id,
            frame_id=data.frame_id,
            timestamp=data.timestamp,
            split=data.split,
            filename=data.filename,
            extra_index_data=data.extra_index_data if data.extra_index_data else None,
        )
        return frame_index_data
