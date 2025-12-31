Overview
========

StandardE2E is a unified framework for preprocessing, indexing, and loading autonomous driving datasets. It enables training models on multiple heterogeneous datasets with a single, consistent API.

Why StandardE2E?
----------------

Training autonomous driving models on diverse datasets is challenging:

- **Different data formats**: Each dataset has unique file structures and APIs
- **Inconsistent modalities**: Camera formats, LiDAR representations, and annotation styles vary
- **Complex preprocessing**: Converting raw data to model-ready format requires dataset-specific code
- **Limited flexibility**: Switching or combining datasets means rewriting data pipelines

StandardE2E solves these problems by providing:

🎯 **One Format to Rule Them All**
   All datasets are converted to a unified :class:`~standard_e2e.data_structures.frame_data.TransformedFrameData` representation with consistent modality keys and metadata.

⚡ **Efficient Indexing**
   Parquet-based indexes enable fast filtering, sampling, and frame lookup without loading heavy scene data.

🔌 **Extensible by Adding New Datasets**
   Once processed, new datasets follow the same API and provide consistent data structures.

🎨 **Flexible Augmentation**
   Chain frame-level augmentations that work across all datasets. Regime-aware (train/val/test).

🔧 **Parametrizable Pipelines**
   Configure what to load programmatically or via YAML config files.
   You can fetch required frames and specific modalities for each of them.
Architecture
------------

.. image:: ../assets/standard_e2e_dataflow.png
   :alt: StandardE2E Data Flow
   :align: center
   :width: 90%

|

StandardE2E consists of two parts:

Preprocessing
-------------

Preprocessing converts raw datasets into a unified, training-ready on-disk format.
It is where dataset-specific processing happens, and where reusable user-defined adapters
are applied.

Preprocessing uses a two-stage data transformation:

1. **Raw dataset →** :class:`~standard_e2e.data_structures.frame_data.StandardFrameData` (dataset-agnostic intermediate format)
      Handled by dataset-specific processors and kept free of user-defined transformations.
      Basically, :class:`~standard_e2e.data_structures.frame_data.StandardFrameData`
      keeps a raw frame data in a consistent structure.
      See `adding_new_dataset.md <https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_ to learn how to add new datasets.


2. :class:`~standard_e2e.data_structures.frame_data.StandardFrameData` → :class:`~standard_e2e.caching.adapters.abstract_adapter.AbstractAdapter` → :class:`~standard_e2e.data_structures.frame_data.TransformedFrameData`.
      Adapters apply user-defined transformations (image resizing, panorama projection, normalization, etc.),
      making the format efficient and training-ready.
      See `intro_tutorial.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/intro_tutorial.ipynb>`_ and `creating_custom_adapter.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/creating_custom_adapter.ipynb>`_ to learn more about adapters.

- Processing stage also produces a Parquet index for fast frame lookup, filtering and
   storing extra metadata.
- As the final stage :class:`~standard_e2e.caching.segment_context.SegmentContextAggregator` may be applied to aggregate
   segment-level context into frame-level data (e.g., current position into trajectory).

Training
--------

Training is where you load the preprocessed data for model training or evaluation.
The main entry point is :class:`~standard_e2e.unified_dataset.UnifiedE2EDataset`,
which accepts an index table from ``index.parquet`` (or from multiple files for combined dataset) and loads frames from
disk. It also handles the following functionalities:

- Filtering the dataset with :class:`~standard_e2e.indexing.filters.index_filter.IndexFilter`,
   eg filtering the frames by some boolean column with :class:`~standard_e2e.indexing.filters.FrameFilterByBooleanColumn`.
- Applying regime-aware augmentations via :class:`~standard_e2e.dataset_utils.augmentation.FrameAugmentation`,
   eg image augmentations with :class:`~standard_e2e.dataset_utils.augmentation.MultipleFramesImageAugmentation`.
- Enforcing modality defaults via :class:`~standard_e2e.dataset_utils.modality_defaults.ModalityDefaults`,
   eg providing :class:`~standard_e2e.enums.Intent.UNKNOWN` intent when intent labels are missing with
   :class:`~standard_e2e.dataset_utils.modality_defaults.IntentDefaults`.
