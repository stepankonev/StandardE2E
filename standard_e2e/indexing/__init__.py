import pandas as pd

from .filters.index_filter import IndexFilter
from .index_data_generator import FrameIndexData, IndexDataGenerator


def get_multi_dataset_index(index_df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple index DataFrames into a single DataFrame.
    """
    if not index_df_list:
        raise ValueError("The list of index DataFrames is empty.")
    for index_df in index_df_list:
        if IndexFilter.FILTERING_COL_NAME not in index_df.columns:
            index_df[IndexFilter.FILTERING_COL_NAME] = True
    concatenated_index = pd.concat(index_df_list, ignore_index=True)
    concatenated_index = concatenated_index.reset_index(drop=True)
    return concatenated_index


__all__ = ["IndexDataGenerator", "FrameIndexData", "get_multi_dataset_index"]
