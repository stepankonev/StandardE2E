from abc import ABC, abstractmethod

from standard_e2e.data_structures import TransformedFrameData


class FrameAugmentation(ABC):
    """Abstract base class for per-frame augmentations.

    Subclasses implement ``_augment`` to transform a mapping of
    ``TransformedFrameData`` keyed by frame name. ``augment`` enforces regime
    validation and delegates to the subclass.
    """

    ALLOWED_REGIMES = ["train", "val", "test"]

    def __init__(self, *args, **kwargs):
        """Construct an augmentation; accepts arbitrary kwargs for subclasses."""

    def augment(
        self, frames: dict[str, TransformedFrameData], regime: str
    ) -> dict[str, TransformedFrameData]:
        """Apply augmentation to the given frame data.

        Args:
            frames (dict[str, TransformedFrameData]): The frame data to augment.
            regime (str): The regime for which the augmentation is applied.

        Returns:
            dict[str, TransformedFrameData]: The augmented frame data.
        """
        if regime not in self.ALLOWED_REGIMES:
            raise ValueError(
                f"Invalid regime: {regime}. Must be one of {self.ALLOWED_REGIMES}."
            )
        return self._augment(frames, regime)

    @abstractmethod
    def _augment(
        self, frames: dict[str, TransformedFrameData], regime: str
    ) -> dict[str, TransformedFrameData]:
        """Apply augmentation to the given frame data.

        Args:
            frames (dict[str, TransformedFrameData]): The frame data to augment.

        Returns:
            dict[str, TransformedFrameData]: The augmented frame data.
        """
        raise NotImplementedError("Subclasses must implement the augment method.")


class IdentityFrameAugmentation(FrameAugmentation):
    """Identity augmentation that returns the frame data unchanged."""

    def _augment(
        self, frames: dict[str, TransformedFrameData], regime: str
    ) -> dict[str, TransformedFrameData]:
        """Return the frame data unchanged."""
        return frames
