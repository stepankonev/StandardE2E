import logging

import pandas as pd

from standard_e2e.indexing.filters.index_filter import IndexFilter


class FrameFilterByBooleanColumn(IndexFilter):
    """
    Filter to select frames based on boolean condition.
    """

    def __init__(self, boolean_column: str, allow_missing_column: bool = False):
        self._boolean_column = boolean_column
        self._allow_missing_column = allow_missing_column
        if self._allow_missing_column:
            logging.warning(
                "Column %s is not required for filtering, "
                "it may lead to unexpected results.",
                self._boolean_column,
            )

    @property
    def name(self) -> str:
        return "frame_boolean_filter"

    def _get_binary_mask(self, index_data: pd.DataFrame) -> pd.Series:
        if self._boolean_column not in index_data.columns:
            if self._allow_missing_column:
                return pd.Series(True, index=index_data.index)
            else:
                raise ValueError(
                    f"Column {self._boolean_column} is required for filtering."
                )
        selector = index_data[self._boolean_column]
        if selector.isnull().any():
            logging.warning(
                "Column %s contains missing values. "
                "Filling missing values with False for filtering.",
                self._boolean_column,
            )
        selector = selector.fillna(False)
        if not pd.api.types.is_bool_dtype(selector):
            raise TypeError(
                f"Column {self._boolean_column} must be of boolean type, "
                f"but found {index_data[self._boolean_column].dtype}."
            )
        return selector
