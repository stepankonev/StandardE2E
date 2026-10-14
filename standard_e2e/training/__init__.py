"""Training: the PyTorch dataset over processed frames, and batching.

This sub-package is the only part of StandardE2E that imports PyTorch. The data
model, the indexing and the preprocessing pipeline do not, so a dataset's
preprocessing environment (``datasets/<name>/``) can leave PyTorch out.
"""

from standard_e2e.training.batching import (
    BatchedCameraData,
    BatchedFrameDetections3D,
    BatchedLidarPointCloud,
    BatchedTrajectory,
    TransformedFrameDataBatch,
    collate_modalities,
)
from standard_e2e.training.multiple_frames_image_augmentation import (
    MultipleFramesImageAugmentation,
)
from standard_e2e.training.unified_dataset import UnifiedE2EDataset

__all__ = [
    "UnifiedE2EDataset",
    "TransformedFrameDataBatch",
    "BatchedTrajectory",
    "BatchedLidarPointCloud",
    "BatchedCameraData",
    "BatchedFrameDetections3D",
    "collate_modalities",
    "MultipleFramesImageAugmentation",
]
