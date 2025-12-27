from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from standard_e2e.data_structures import (
    TransformedFrameData,
    TransformedFrameDataBatch,
)
from standard_e2e.dataset_utils.augmentation import (
    FrameAugmentation,
    IdentityFrameAugmentation,
)
from standard_e2e.dataset_utils.frame_loader import (
    FrameLoader,
    create_frame_loaders_from_config,
)
from standard_e2e.dataset_utils.modality_defaults import (
    ModalityDefaults,
    _check_modality_defaults_dict,
)
from standard_e2e.enums import Modality
from standard_e2e.indexing.filters import IndexFilter


class UnifiedE2EDataset(Dataset):
    """
    UnifiedE2EDataset wraps feature and label frame loaders, applies optional
    index filters and augmentations, and provides PyTorch-compatible access to
    transformed frames for end-to-end training or evaluation.

    Parameters
    ----------
    index_data : pd.DataFrame
        Index dataframe describing available frames and metadata used by frame loaders.
    processed_data_path : str
        Root path where processed feature/label frames are stored.
    regime : str
        Augmentation regime; must be one of FrameAugmentation.ALLOWED_REGIMES.
    feature_loaders : list[FrameLoader] | None, optional
        Pre-instantiated feature frame loaders; mutually exclusive with
        feature_loaders_config.
    label_loaders : list[FrameLoader] | None, optional
        Pre-instantiated label frame loaders; mutually exclusive with
        label_loaders_config.
    feature_loaders_config : dict | list[dict] | None, optional
        Configuration(s) to build feature frame loaders when feature_loaders is None.
    label_loaders_config : dict | list[dict] | None, optional
        Configuration(s) to build label frame loaders when label_loaders is None.
    augmentations : list[FrameAugmentation] | None, optional
        Ordered list of frame augmentations to apply; defaults to IdentityFrameAugmentation.
    index_filters : list[IndexFilter] | None, optional
        Filters applied to index_data to select usable rows.
    modality_defaults : dict[Modality, ModalityDefaults] | None, optional
        Default modality-specific settings passed to frame loaders.
    """

    def __init__(
        self,
        index_data: pd.DataFrame,
        processed_data_path: str,
        regime: str,
        feature_loaders: list[FrameLoader] | None = None,
        label_loaders: list[FrameLoader] | None = None,
        feature_loaders_config: dict | list[dict] | None = None,
        label_loaders_config: dict | list[dict] | None = None,
        augmentations: list[FrameAugmentation] | None = None,
        index_filters: list[IndexFilter] | None = None,
        modality_defaults: dict[Modality, ModalityDefaults] | None = None,
    ):
        if regime not in FrameAugmentation.ALLOWED_REGIMES:
            raise ValueError(
                f"Invalid regime: {regime}. Must be one of \
                    {FrameAugmentation.ALLOWED_REGIMES}."
            )
        _check_modality_defaults_dict(modality_defaults)
        self._regime = regime
        self._index_data = index_data.copy()
        self._index_filters = index_filters
        self._apply_index_filters()
        self._iloc_to_idx = self._index_data[
            self._index_data[IndexFilter.FILTERING_COL_NAME]
        ].index
        self._processed_data_path = processed_data_path
        self._frame_loaders = self._create_feature_and_label_loaders(
            feature_loaders,
            label_loaders,
            feature_loaders_config,
            label_loaders_config,
        )
        self._augmentations = augmentations or [IdentityFrameAugmentation()]
        self._modality_defaults = modality_defaults

    def _apply_index_filters(self):
        if self._index_filters:
            for index_filter in self._index_filters:
                self._index_data = index_filter.filter(self._index_data)
        if not self._index_filters:
            self._index_data[IndexFilter.FILTERING_COL_NAME] = True

    def _create_feature_and_label_loaders(
        self,
        feature_loaders: list[FrameLoader] | None,
        label_loaders: list[FrameLoader] | None,
        feature_loaders_config: dict | list[dict] | None,
        label_loaders_config: dict | list[dict] | None,
    ) -> list[FrameLoader]:
        if feature_loaders is None and feature_loaders_config is None:
            raise ValueError(
                "Both feature_loaders and feature_loaders_config are not defined."
            )
        if label_loaders is None and label_loaders_config is None:
            raise ValueError(
                "Both label_loaders and label_loaders_config are not defined."
            )
        if feature_loaders is not None and feature_loaders_config is not None:
            raise ValueError(
                "Both feature_loaders and feature_loaders_config are defined."
            )
        if label_loaders is not None and label_loaders_config is not None:
            raise ValueError("Both label_loaders and label_loaders_config are defined.")

        if feature_loaders:
            for loader in feature_loaders:
                loader.set_index_data(self._index_data)
        else:
            if feature_loaders_config is None:
                raise ValueError("feature_loaders_config must not be None here")
            fl_config: list[dict]
            if isinstance(feature_loaders_config, dict):
                fl_config = [feature_loaders_config]
            elif isinstance(feature_loaders_config, list):
                fl_config = feature_loaders_config
            else:
                raise TypeError("feature_loaders_config must be dict or list of dict")
            feature_loaders = create_frame_loaders_from_config(
                processed_data_path=self._processed_data_path,
                frames_description=fl_config,
                location="features",
                index_data=self._index_data,
            )

        if label_loaders:
            for loader in label_loaders:
                loader.set_index_data(self._index_data)
        else:
            if label_loaders_config is None:
                raise ValueError("label_loaders_config must not be None here")
            ll_config: list[dict]
            if isinstance(label_loaders_config, dict):
                ll_config = [label_loaders_config]
            elif isinstance(label_loaders_config, list):
                ll_config = label_loaders_config
            else:
                raise TypeError("label_loaders_config must be dict or list of dict")
            label_loaders = create_frame_loaders_from_config(
                processed_data_path=self._processed_data_path,
                frames_description=ll_config,
                location="labels",
                index_data=self._index_data,
            )

        result = feature_loaders + label_loaders
        for loader in result:
            if loader.processed_data_path is None:
                loader.set_processed_data_path(self._processed_data_path)
        return result

    def __len__(self) -> int:
        return int(self._index_data[IndexFilter.FILTERING_COL_NAME].sum())

    def _fetch_frames(self, current_frame_idx: int) -> dict[str, TransformedFrameData]:
        frames: dict[str, TransformedFrameData] = {}
        for frame_loader in self._frame_loaders:
            frame = frame_loader.load_frame(
                current_frame_idx=current_frame_idx,
                index_data=self._index_data,
                modality_defaults=self._modality_defaults,
            )
            frames.update(frame)
        return frames

    def _augment_frames(
        self, frames: dict[str, TransformedFrameData]
    ) -> dict[str, TransformedFrameData]:
        for augmentation in self._augmentations:
            frames = augmentation.augment(frames, regime=self._regime)
        return frames

    def __getitem__(self, requested_ord_idx: int) -> dict[str, TransformedFrameData]:
        current_frame_idx = self._iloc_to_idx[requested_ord_idx]
        frames: dict[str, TransformedFrameData] = self._fetch_frames(current_frame_idx)
        frames = self._augment_frames(frames)
        return frames

    @classmethod
    def collate_fn(
        cls,
        batch: list[dict[str, TransformedFrameData]],
        device: Optional[torch.device] = None,
    ) -> dict[str, TransformedFrameDataBatch]:
        """
        Collate a list of {key: TransformedFrameData} into \
            {key: TransformedFrameDataBatch}.
        TransformedFrameDataBatch internally uses PyTorch's \
            collate with a Trajectory override.
        """
        device = device or torch.device("cpu")
        collated: Dict[str, TransformedFrameDataBatch] = {}
        keys = batch[0].keys()
        for key in keys:
            collated[key] = TransformedFrameDataBatch(
                [sample[key] for sample in batch], device=device
            )
        return collated
