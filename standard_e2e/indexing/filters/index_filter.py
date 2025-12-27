import logging
from abc import ABC, abstractmethod

import pandas as pd


class IndexFilter(ABC):
    """Class to filter index data based on specific criteria."""

    FILTERING_COL_NAME = "_FILTERING_COL"

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the filter."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_binary_mask(self, index_data: pd.DataFrame) -> pd.Series:
        """Generate binary mask for filtering."""
        raise NotImplementedError("Subclasses must implement this method.")

    def filter(self, index_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to the index data."""
        if not isinstance(index_data, pd.DataFrame):
            raise TypeError("index_data must be a pandas DataFrame")
        if index_data.empty:
            logging.warning("Index data is empty. No filtering applied.")
            return index_data
        if self.FILTERING_COL_NAME not in index_data.columns:
            index_data[self.FILTERING_COL_NAME] = True

        initial_size = index_data[self.FILTERING_COL_NAME].sum()
        current_binary_mask = self._get_binary_mask(index_data)
        index_data[self.FILTERING_COL_NAME] = (
            index_data[self.FILTERING_COL_NAME] & current_binary_mask
        )
        final_size = index_data[self.FILTERING_COL_NAME].sum()

        logging.info(
            "Filter '%s' applied: %d -> %d rows", self.name, initial_size, final_size
        )
        logging.info("New to old ratio: %.2f", final_size / initial_size)
        if final_size == 0:
            logging.warning("No data left after filtering.")
        return index_data
