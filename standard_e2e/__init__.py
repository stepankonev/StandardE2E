"""StandardE2E — a unified framework for end-to-end autonomous-driving datasets.

The most-used public types are re-exported here for convenience, so user code
and notebooks can do, e.g.::

    from standard_e2e import TransformedFrameData, Modality, TrajectoryComponent

The original submodule paths (``standard_e2e.data_structures``,
``standard_e2e.enums``, ...) keep working unchanged — these are aliases.
"""

__version__ = "0.0.7"
__author__ = "stepankonev"

from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
    HDMap,
    LidarData,
    LidarPointCloud,
    MapElement,
    StandardFrameData,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    Intent,
    LidarComponent,
    MapElementType,
    Modality,
    TrajectoryComponent,
)
from standard_e2e.unified_dataset import UnifiedE2EDataset

__all__ = [
    "__version__",
    "__author__",
    # training / loading
    "UnifiedE2EDataset",
    # frame containers
    "StandardFrameData",
    "TransformedFrameData",
    "CameraData",
    "LidarData",
    "LidarPointCloud",
    "HDMap",
    "MapElement",
    "Detection3D",
    "FrameDetections3D",
    "Trajectory",
    # enums
    "Modality",
    "TrajectoryComponent",
    "CameraDirection",
    "Intent",
    "DetectionType",
    "MapElementType",
    "LidarComponent",
]
