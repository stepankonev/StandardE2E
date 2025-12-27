import albumentations as A
from albumentations.pytorch import ToTensorV2

from standard_e2e.data_structures import TransformedFrameData
from standard_e2e.dataset_utils.augmentation import FrameAugmentation
from standard_e2e.enums import Modality


class MultipleFramesImageAugmentation(FrameAugmentation):
    """Apply an Albumentations pipeline to each camera modality in given frames."""

    def __init__(
        self,
        regime: str,
        train_transform: A.Compose | None = None,
        val_transform: A.Compose | None = None,
        *args,
        **kwargs,
    ):
        """Create an augmentation with train/val transforms.

        Args:
            regime: Regime used to pick between train/val transforms.
            train_transform: Optional Albumentations compose for training.
            val_transform: Optional Albumentations compose for validation/test.
            *args, **kwargs: Passed to ``FrameAugmentation``.
        """
        super().__init__(*args, **kwargs)
        self._regime = regime
        if train_transform is not None:
            self._train_transform = train_transform
        else:
            self._train_transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, p=1),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Rotate(limit=5, p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                    A.Normalize(),
                    ToTensorV2(),
                ],
            )
        if val_transform is not None:
            self._val_transform = val_transform
        else:
            self._val_transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, p=1),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        self._transform = (
            self._train_transform if regime == "train" else self._val_transform
        )

    def _augment(
        self, frames: dict[str, TransformedFrameData], regime: str
    ) -> dict[str, TransformedFrameData]:
        """Apply augmentation to the given frame data."""
        frame_keys_list = [
            key
            for key in frames.keys()
            if Modality.CAMERAS in frames[key].get_present_modality_keys()
        ]
        result_images = self._transform(
            images=[
                frames[k].get_modality_data(Modality.CAMERAS)
                for k in frame_keys_list
                if Modality.CAMERAS in frames[k].get_present_modality_keys()
            ]
        )
        for k, frame_name in enumerate(frame_keys_list):
            frames[frame_name].set_modality_data(
                Modality.CAMERAS, result_images["images"][k]
            )
        return frames
