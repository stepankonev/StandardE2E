import pandas as pd

from standard_e2e.dataset_utils.selector.frame_selector import FrameSelector


class CurrentSelector(FrameSelector):
    """Selector that always returns the current frame (no offset)."""

    @property
    def name(self) -> str:
        return "current_selector"

    def select_frame(
        self, current_frame_iloc: int, index_data: pd.DataFrame | None = None
    ) -> int:
        """Return the current frame index unchanged.

        Args:
            current_frame_iloc: Row position (iloc) of the reference frame.
            index_data: Unused; present for interface consistency.
        """
        return current_frame_iloc
