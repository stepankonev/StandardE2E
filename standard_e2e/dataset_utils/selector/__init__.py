import pandas as pd

from standard_e2e.dataset_utils.selector.closest_timestamp_selector import (
    ClosestTimestampSelector,
)
from standard_e2e.dataset_utils.selector.current_selector import CurrentSelector
from standard_e2e.dataset_utils.selector.frame_selector import FrameSelector


def create_frame_selector_from_config(
    config, location, index_data: pd.DataFrame | None
) -> FrameSelector:
    """Factory to build a ``FrameSelector`` from a YAML/JSON-like config.

    Args:
        config: Mapping with ``name`` and ``parameters`` keys describing selector.
        location: "features" or "labels"; forwarded to the selector ctor.
        index_data: Optional index dataframe for eager precomputation.
    """
    if config["name"] == "closest_timestamp":
        return ClosestTimestampSelector(
            location=location,
            index_data=index_data,
            delta_t=config["parameters"].get("delta_t"),
        )
    elif config["name"] == "current":
        return CurrentSelector(location=location)
    else:
        raise ValueError(f"Unknown frame selector type: {config['name']}")


__all__ = [
    "ClosestTimestampSelector",
    "FrameSelector",
    "CurrentSelector",
    "create_frame_selector_from_config",
]
