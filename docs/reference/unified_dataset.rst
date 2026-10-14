Unified Dataset
================

The PyTorch side of StandardE2E lives in :mod:`standard_e2e.training`, the only
sub-package that imports PyTorch.

.. currentmodule:: standard_e2e.training

.. autoclass:: UnifiedE2EDataset
   :members:
   :undoc-members:
   :show-inheritance:

Batching
--------

.. autosummary::
   :toctree: generated/training
   :nosignatures:

   TransformedFrameDataBatch
   BatchedTrajectory
   BatchedLidarPointCloud
   BatchedCameraData
   BatchedFrameDetections3D
   collate_modalities
