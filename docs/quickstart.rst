Quickstart
==========

Welcome to StandardE2E! This guide will get you up and running quickly.

Installation
------------

The default environment is the training one: StandardE2E with PyTorch and no dataset
SDKs. Install it from PyPI:

.. code-block:: bash

   pip install standard-e2e

Or work from source with `uv <https://docs.astral.sh/uv/>`_ (``--all-extras`` adds the
test and lint tools):

.. code-block:: bash

   git clone https://github.com/stepankonev/StandardE2E.git
   cd StandardE2E
   uv sync --all-extras

Preprocessing (required before training)
----------------------------------------

Raw data is converted into the unified format once per dataset: scene files containing
:class:`~standard_e2e.data_structures.frame_data.TransformedFrameData` and the index used
by the :class:`~standard_e2e.training.UnifiedE2EDataset`. Each dataset is preprocessed in
its own isolated environment, which holds only the SDK that dataset needs (TensorFlow
for Waymo, ``av2`` for Argoverse 2, nuplan-devkit for NAVSIM, ...) and no PyTorch. Its
folder in the repository, ``datasets/<name>/``, carries that environment and a
preprocessing pipeline that `stepix <https://github.com/stepankonev/stepix>`_ runs: set
the paths in the ``[params]`` block of ``datasets/<name>/README.md``, then run, from the
repository root:

.. code-block:: bash

   uvx stepix run datasets/waymo_e2e/README.md

The pipeline checks the input, extracts archives where the dataset ships them, and
runs the converter in the dataset's environment. The same converter can be run by hand
in that environment, e.g. for the Waymo End-to-End training split:

.. code-block:: bash

   uv run --project datasets/waymo_e2e python -m standard_e2e.caching.process_source_dataset waymo_e2e \
       --input_path=path/to/input \
       --output_path=path/to/output \
       --split=training \
       --num_workers=32 \
       --config_file=configs/base.yaml \
       --do_parallel_processing

See the `datasets/ <https://github.com/stepankonev/StandardE2E/tree/main/datasets>`_
folder for every supported dataset's description, input layout and pipeline, and the
in-repo examples for the programmatic flow:

- Script: `examples/dataset_preprocessing.py <https://github.com/stepankonev/StandardE2E/blob/main/examples/dataset_preprocessing.py>`_
- Notebook walkthrough: `notebooks/intro_tutorial.ipynb <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/intro_tutorial.ipynb>`_

For the NATIX Edge Case dataset (``natix_edgecase``; the curated
edge-case sibling of ``natix_multicam`` — see :doc:`datasets`) the
invocation is the standard one over the downloaded Hugging Face tree,
but the release's **VLM event annotations** deserve a note on where
they end up. Every emitted frame stores the events covering its
timestamp inside its ``.npz`` as ``aux_data["edge_case_events"]`` — one
plain dict per covering event carrying the verbatim ``label``, the
``start_sec`` / ``end_sec`` window (video seconds from clip start) and
the complete ``ai_analysis`` object (event classification, visual
evidence, contextual analysis, agentic validation, detected event); the
list is empty outside events. Two flattened index columns make in-event
frames selectable without opening any ``.npz``:

.. code-block:: python

   import numpy as np
   import pandas as pd

   root = "path/to/output"
   index = pd.read_parquet(f"{root}/natix_edgecase/all/index.parquet")

   # Select frames covered by an annotated event (2908 of 10747 in the
   # first release) straight from the index...
   in_event = index[index.extra_edge_case_count > 0]
   print(in_event.extra_edge_case.iloc[0])   # summary text

   # ...and read the full VLM payload from the frame's npz aux data.
   frame = np.load(f"{root}/{in_event.filename.iloc[0]}", allow_pickle=True)
   events = frame["aux_data"].item()["edge_case_events"]
   print(events[0]["ai_analysis"]["EVENT CLASSIFICATION"])

The same ``aux_data`` dict is available on every frame loaded through
:class:`~standard_e2e.training.UnifiedE2EDataset` (as
``frame.aux_data``), so the annotations ride along into training
pipelines with no extra plumbing.

.. tip::
   ``--num_workers`` and ``--do_parallel_processing`` cover **both**
   pipeline stages: the per-frame conversion (one frame per worker) and
   the per-segment context aggregation (one segment per worker). The
   aggregator stage runs after frame conversion completes; the same
   pool size is reused.

.. tip::
   For quicker debug runs, set ``STANDARD_E2E_DEBUG=true``. Some datasets may
   truncate segment continuity in this mode, so use only for smoke-testing.

Visualizing processed output
----------------------------

Any processed ``<dataset>/<split>`` folder (the ``.npz`` frames plus
``index.parquet`` written above) can be rendered to per-scene MP4s with one
dataset-agnostic tool -- it auto-detects whichever modalities each frame
carries: a camera mosaic (a per-direction surround grid, or a stitched
panorama as the pano adapter emits) and a single co-registered BEV panel
(HD-map / lidar / detection rasters, ``lidar_pc`` points, vector 3D boxes,
past / future / preference trajectories, ego).

.. code-block:: bash

   uv run python -m standard_e2e.visualization.visualize_processed \
       path/to/output/DATASET_NAME/SPLIT --num-scenes 2 --out visualizations

Select scenes with ``--num-scenes N`` or repeatable ``--scene-id ID`` (default:
the first scene); cap length with ``--max-frames``. Playback speed defaults to
the rate inferred from the frame timestamps, so videos play in real time (very
low rates are encoded by duplicating frames, keeping the real-time duration);
pass ``--fps`` to override. The BEV meter grids and channel orders are read
from the ``dataset_info.yaml`` / per-frame metadata the converter writes, so
the panel is correct for any adapter grid configuration without per-dataset
code.

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

      uv run --project datasets/waymo_e2e python examples/dataset_preprocessing.py \
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
