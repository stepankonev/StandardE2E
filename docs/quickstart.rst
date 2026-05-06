Quickstart
==========

Welcome to StandardE2E! This guide will get you up and running quickly.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install standard-e2e

Or from source for development (with uv):

.. code-block:: bash

   git clone https://github.com/stepankonev/StandardE2E.git
   cd StandardE2E
   uv sync --all-extras

Or with pip/conda: create a venv, then ``pip install -e ".[dev]"``.

Preprocessing (required before training)
----------------------------------------

Convert raw data into the unified format before any training or loading. This
Waymo End-to-End example (training split) builds scene files containing
:class:`~standard_e2e.data_structures.frame_data.TransformedFrameData` and the index used by the :class:`~standard_e2e.unified_dataset.UnifiedE2EDataset`.

.. code-block:: bash

   uv run python -m standard_e2e.caching.process_source_dataset waymo_e2e \
       --input_path=path/to/input \
       --output_path=path/to/output \
       --split=training \
       --num_workers=32 \
       --config_file=path/to/config.yaml \
       --do_parallel_processing

See the in-repo examples for the exact flow:
- Script: `examples/dataset_preprocessing.py <https://github.com/stepankonev/StandardE2E/blob/main/examples/dataset_preprocessing.py>`_
- Notebook walkthrough: `notebooks/intro_tutorial.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/intro_tutorial.ipynb>`_

For Argoverse 2 Sensor the equivalent invocation is:

.. code-block:: bash

   uv run python -m standard_e2e.caching.process_source_dataset av2_sensor \
       --input_path=path/to/argoverse2/sensor \
       --output_path=path/to/output \
       --split=train \
       --num_workers=32 \
       --config_file=path/to/config.yaml \
       --do_parallel_processing

A ready-made wrapper for AV2 Sensor lives at
`scripts/prepare_dataset_av2_sensor.sh <https://github.com/stepankonev/StandardE2E/blob/main/scripts/prepare_dataset_av2_sensor.sh>`_.

For Argoverse 2 Lidar (same on-disk format as Sensor minus cameras and
``annotations.feather``; ~20× more logs but no 3D box labels) the
invocation is the same with ``av2_lidar`` and an input path pointing at
the lidar split:

.. code-block:: bash

   uv run python -m standard_e2e.caching.process_source_dataset av2_lidar \
       --input_path=path/to/argoverse2/lidar \
       --output_path=path/to/output \
       --split=train \
       --num_workers=32 \
       --config_file=path/to/config.yaml \
       --do_parallel_processing

The processor inherits from :class:`~standard_e2e.caching.src_datasets.av2_sensor.Av2SensorDatasetProcessor`
and short-circuits the camera + detection helpers so the missing
modalities surface as defaults at training time via
:class:`~standard_e2e.dataset_utils.modality_defaults.ModalityDefaults`.

.. tip::
   For quicker debug runs, set ``STANDARD_E2E_DEBUG=true``. Some datasets may
   truncate segment continuity in this mode, so use only for smoke-testing.

Interactive Tutorials
---------------------

The best way to learn StandardE2E is through our interactive Jupyter notebooks:

1. **Introduction Tutorial** (`intro_tutorial.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/intro_tutorial.ipynb>`_)
   
   Your first steps with StandardE2E. Learn the basic concepts, data structures, and how to load your first dataset.

2. **Data Containers** (`containers.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/containers.ipynb>`_)
   
   Deep dive into StandardE2E's data containers: cameras, LiDAR, trajectories, and detections. Understand how multimodal data is represented.

3. **Multi-Dataset Training** (`multi_dataset_training_and_filtering.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/multi_dataset_training_and_filtering.ipynb>`_)
   
   Learn how to train models on multiple datasets simultaneously and apply powerful filtering strategies.

4. **Creating Custom Adapters** (`creating_custom_adapter.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/creating_custom_adapter.ipynb>`_)
   
   Extend StandardE2E by creating your own adapters for custom preprocessing and augmentation pipelines.

Code Examples
-------------

For complete working examples, see:

1. **Dataset Preprocessing** - `dataset_preprocessing.py <https://github.com/stepankonev/StandardE2E/blob/main/examples/dataset_preprocessing.py>`_

   .. code-block:: bash

      uv run python examples/dataset_preprocessing.py \
          --e2e_dataset_path /path/to/raw/waymo/e2e \
          --split training \
          --processed_data_path /path/to/processed

2. **Simple Training Loop** - `very_simple_training.py <https://github.com/stepankonev/StandardE2E/blob/main/examples/very_simple_training.py>`_

   .. code-block:: bash

      uv run python examples/very_simple_training.py \
          --processed_data_path /path/to/processed

3. **Multi-Dataset DataLoader** - `creating_unified_dataloader.py <https://github.com/stepankonev/StandardE2E/blob/main/examples/creating_unified_dataloader.py>`_

   .. code-block:: bash

      uv run python examples/creating_unified_dataloader.py \
          --processed_data_path /path/to/processed

Adding New Datasets
-------------------

Want to add support for your own dataset? Check out our comprehensive guide:

📖 `Adding New Datasets Guide <https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_

This guide covers:

- Dataset processor architecture
- Implementing custom converters
- Segment context aggregation
- Integration with the preprocessing pipeline

Next Steps
----------
- 🔍 Explore the :doc:`overview` to understand the architecture
- 📖 Check the :doc:`reference/api` for complete API documentation
- 💡 Browse `examples/ <https://github.com/stepankonev/StandardE2E/tree/main/examples>`_ and the notebooks for tested flows

Need Help?
----------

- 🐛 Report issues on `GitHub Issues <https://github.com/stepankonev/StandardE2E/issues>`_
- 📧 Reach out to the maintainers in `Discord <https://discord.gg/vJnQNcQGQ8>`_

Happy coding! 🚗💨
