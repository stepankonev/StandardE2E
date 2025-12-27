User Guide
==========

This guide covers installation, core concepts, configuration, and common workflows for StandardE2E.

.. note::
   New to StandardE2E? Start with the :doc:`quickstart` guide first!

Installation
------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release:

.. code-block:: bash

   pip install standard-e2e

From Source (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors or to use the latest features:

.. code-block:: bash

   git clone https://github.com/stepankonev/StandardE2E.git
   cd StandardE2E
   conda create -n standard_e2e python=3.12
   conda activate standard_e2e
   pip install -e .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For specific datasets or features:

.. code-block:: bash

   # For Waymo dataset support
   pip install waymo-open-dataset-tf-2-12-0
   
   # For advanced augmentations
   pip install albumentations
   
   # For visualization tools
   pip install matplotlib plotly

Core Concepts
-------------

This section explains the fundamental building blocks of StandardE2E.

Index Data
~~~~~~~~~~

The **index** is a Parquet file (see ``standard_e2e.constants.INDEX_FILE_NAME``) that stores frame-level metadata:

- ``frame_id``: Unique identifier for each frame
- ``segment_id``: Which driving segment/scene the frame belongs to
- ``timestamp``: Frame timestamp
- ``dataset_name``: Source dataset (e.g., "waymo_e2e", "argoverse2")
- Additional metadata: weather, time of day, location, etc.

The index enables:

- **Fast filtering**: Query frames by attributes without loading data
- **Efficient sampling**: Random access to specific frames
- **Multi-dataset alignment**: Common schema across datasets

.. code-block:: python

   import pandas as pd
   
   # Load index
   index = pd.read_parquet("/path/to/processed/index.parquet")
   
   # Filter to daytime urban scenes
   filtered = index[
       (index['time_of_day'] == 'day') & 
       (index['location_type'] == 'urban')
   ]
   
   print(f"Found {len(filtered)} matching frames")

Frame Loaders
~~~~~~~~~~~~~

**FrameLoader** objects map configuration entries to actual frame files:

- Takes a frame specification (name + modalities + selector)
- Locates and loads the corresponding ``.npz`` file from ``processed_data_path``
- Returns a ``TransformedFrameData`` object

Frame loaders are created from config:

.. code-block:: python

   from standard_e2e.dataset_utils.frame_loader import create_frame_loaders_from_config
   from standard_e2e.enums import Modality
   
   # Modalities are specified as strings in config
   # Available: cameras, lidar_bev, lidar_pc, hd_map, speed, intent,
   #            future_states, past_states, preference_trajectory, detections_3d
   
   from standard_e2e.dataset_utils.selector import CurrentSelector
   from standard_e2e.dataset_utils.frame_loader import FrameLoader
   from standard_e2e.enums import Modality
   
   loader = FrameLoader(
       frame_name="current",
       required_modalities=[Modality.CAMERAS, Modality.LIDAR_PC],
       frame_selector=CurrentSelector(location="features"),
   )

Selectors
~~~~~~~~~

**FrameSelector** determines which frame to fetch relative to the current index row:

- ``current``: Returns the indexed frame itself (``CurrentSelector``)
- ``closest_timestamp``: Fetches the frame with closest timestamp to a given time delta
  
  - Example: ``{"name": "closest_timestamp", "parameters": {"delta_t": 1.0}}`` → 1 second in the future
  - For features: delta_t must be ≤ 0 (past or current)
  - For labels: delta_t must be ≥ 0 (current or future)
  
- Custom selectors: Extend ``FrameSelector`` base class

.. code-block:: python

   # Current frame
   {
       "frame_name": "current",
       "modalities": ["cameras"],
       "selector": {"name": "current"}
   }
   
   # Future frame (1 second ahead)
   {
       "frame_name": "future_1s",
       "modalities": ["cameras"],
       "selector": {
           "name": "closest_timestamp",
           "parameters": {"delta_t": 1.0}
       }
   }

Augmentations
~~~~~~~~~~~~~

**FrameAugmentation** subclasses transform loaded frames:

- Applied during data loading in training pipeline
- Regime-aware: different augmentations for train/val/test
- ``IdentityFrameAugmentation``: default no-op augmentation

Common augmentations:

- Geometric: rotation, scaling, flipping
- Photometric: color jitter, brightness, contrast
- Spatial: crop, resize

.. code-block:: python

   from standard_e2e.dataset_utils.augmentation import IdentityFrameAugmentation
   
   # Custom augmentation (simplified example)
   class MyAugmentation(IdentityFrameAugmentation):
       def augment_frame(self, frame_data, regime):
           if regime == "train":
               # Apply training-time augmentations
               frame_data = self.apply_transforms(frame_data)
           return frame_data

Modality Defaults
~~~~~~~~~~~~~~~~~

**ModalityDefaults** fills or normalizes missing modality data:

- Handles cases where some frames lack certain modalities
- Provides default values or placeholders
- Ensures consistent batch shapes

.. code-block:: python

   from standard_e2e.dataset_utils.modality_defaults import ModalityDefaults
   from standard_e2e.enums import Modality
   
   # Define defaults for missing camera data
   modality_defaults = {
       Modality.CAMERA: ModalityDefaults(
           use_defaults=True,
           fill_value=0.0
       )
   }

Configuration
-------------

The YAML config (see ``configs/base.yaml``) defines the data pipeline:


Preprocessing Section
~~~~~~~~~~~~~~~~~~~~~

Defines adapters and transforms for converting raw data:

.. code-block:: yaml

   preprocessing:
     adapters:
       - name: camera_resize
         type: ResizeAdapter
         params:
           target_size: [512, 512]
       - name: lidar_filter
         type: LiDARFilterAdapter
         params:
           max_range: 80.0

Dataset Features Section
~~~~~~~~~~~~~~~~~~~~~~~~~

Specifies model inputs:

.. code-block:: yaml

   dataset:
     features:
       - frame_name: current_sensors
         modalities: [cameras, lidar_pc, intent, past_states]
         selector:
           name: current
       
       - frame_name: past_1s
         modalities: [cameras]
         selector:
           name: closest_timestamp
           parameters:
             delta_t: -1.0

Dataset Labels Section
~~~~~~~~~~~~~~~~~~~~~~

Specifies supervision signals:

.. code-block:: yaml

   dataset:
     labels:
       - frame_name: current_future_states
         modalities: [future_states, preference_trajectory]
         selector:
           name: current
       
       - frame_name: future_1second
         modalities: [future_states]
         selector:
           name: closest_timestamp
           parameters:
             delta_t: 1.0

Index Filters
~~~~~~~~~~~~~

Optional filters to include/exclude frames:

.. code-block:: yaml

   dataset:
     index_filters:
       - type: time_of_day
         values: [day, dusk]
       - type: weather
         values: [clear, cloudy]
       - type: min_trajectory_length
         value: 50

Complete Example
~~~~~~~~~~~~~~~~

Full configuration example:

.. code-block:: yaml

   preprocessing:
     adapters:
       - name: resize_images
         type: ResizeAdapter
   
   dataset:
     features:
       - frame_name: current
         modalities: [CAMERA, LIDAR]
         selector: {name: current_frame}
     
     labels:
       - frame_name: trajectory
         modalities: [FUTURE_STATES]
         selector: {name: current_frame}
     
     index_filters:
       - type: dataset_name
         values: [waymo_e2e]

Loading a Dataset
-----------------

Complete example of loading and using a dataset:

.. code-block:: python

   import pandas as pd
   from torch.utils.data import DataLoader
   from standard_e2e import UnifiedE2EDataset
   from standard_e2e.dataset_utils.frame_loader import FrameLoader
   from standard_e2e.dataset_utils.selector import CurrentSelector, ClosestTimestampSelector
   from standard_e2e.enums import Modality

   # 1. Load the index
   index = pd.read_parquet("/path/to/processed/index.parquet")
   
   # Optional: Filter the index
   index = index[index['dataset_name'] == 'waymo_e2e']

   # 2. Create feature loaders (explicit API)
   feature_loaders = [
       FrameLoader(
           frame_name="current_sensors",
           required_modalities=[Modality.CAMERAS, Modality.INTENT, Modality.PAST_STATES],
           frame_selector=CurrentSelector(location="features"),
       )
   ]

   # 3. Create label loaders
   label_loaders = [
       FrameLoader(
           frame_name="current_future_states",
           required_modalities=[Modality.FUTURE_STATES],
           frame_selector=CurrentSelector(location="labels"),
       ),
       FrameLoader(
           frame_name="future_1second",
           required_modalities=[Modality.CAMERAS],
           frame_selector=ClosestTimestampSelector(location="labels", delta_t=1.0),
       ),
   ]

   # 4. Create the dataset
   dataset = UnifiedE2EDataset(
       index_data=index,
       processed_data_path="/path/to/processed",
       regime="train",  # or "val", "test"
       feature_loaders=feature_loaders,
       label_loaders=label_loaders,
   )

   # 5. Create DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=4,
       shuffle=True,
       num_workers=4,
       collate_fn=UnifiedE2EDataset.collate_fn,
       pin_memory=True,
   )

   # 6. Iterate through batches
   for batch_idx, batch in enumerate(dataloader):
       # batch is a dict with frame names as keys
       current_sensors = batch["current_sensors"]
       future_states = batch["current_future_states"]
       
       # Access specific modalities using get_modality_data
       cameras = current_sensors.get_modality_data(Modality.CAMERAS)
       intent = current_sensors.get_modality_data(Modality.INTENT)
       past_states = current_sensors.get_modality_data(Modality.PAST_STATES)
       future = future_states.get_modality_data(Modality.FUTURE_STATES)
       
       # Your training logic here
       print(f"Batch {batch_idx}: {cameras.images.shape}")
       break

Working with Batches
--------------------

Understanding batch structure:

.. code-block:: python

   # Batch is a dict with frame names as keys
   batch = {
       "current_sensors": TransformedFrameData,
       "current_future_states": TransformedFrameData,
       # ... other frames defined in config
   }
   
   # Each TransformedFrameData contains modality data
   frame_data = batch["current_sensors"]
   # Access via get_modality_data(Modality)
   cameras = frame_data.get_modality_data(Modality.CAMERAS)
   # CamerasBatch with:
   #   - images: torch.Tensor (B, N_cameras, H, W, C)
   #   - intrinsics: torch.Tensor (B, N_cameras, 3, 3)
   #   - extrinsics: torch.Tensor (B, N_cameras, 4, 4)

Accessing batch data:

.. code-block:: python

   # Camera images
   cameras = batch["current_sensors"].get_modality_data(Modality.CAMERAS)
   camera_images = cameras.images
   # Shape: (batch_size, num_cameras, height, width, channels)
   
   # LiDAR point cloud
   lidar_pc = batch["current_sensors"].get_modality_data(Modality.LIDAR_PC)
   # LiDARBatch object
   
   # Future trajectory
   future_states = batch["current_future_states"].get_modality_data(Modality.FUTURE_STATES)
   # BatchedTrajectory object with .get([components]) method

Common Workflows
----------------

Multi-Dataset Training
~~~~~~~~~~~~~~~~~~~~~~

Combine multiple datasets:

.. code-block:: python

   # Load indexes
   waymo_idx = pd.read_parquet("/data/waymo/index.parquet")
   argo_idx = pd.read_parquet("/data/argoverse/index.parquet")
   
   # Add dataset identifier if not present
   waymo_idx['source'] = 'waymo'
   argo_idx['source'] = 'argoverse'
   
   # Concatenate
   combined_idx = pd.concat([waymo_idx, argo_idx], ignore_index=True)
   
   # Shuffle for better mixing
   combined_idx = combined_idx.sample(frac=1.0, random_state=42)
   
   # Create dataset as usual
   dataset = UnifiedE2EDataset(combined_idx, ...)

Temporal Context Loading
~~~~~~~~~~~~~~~~~~~~~~~~~

Load past and future frames:

.. code-block:: python

   from standard_e2e.dataset_utils.selector import CurrentSelector, ClosestTimestampSelector
   from standard_e2e.dataset_utils.frame_loader import FrameLoader
   from standard_e2e.enums import Modality
   
   # Can use FrameLoader objects directly
   feature_loaders = [
       # Current frame
       FrameLoader(
           frame_name="current",
           required_modalities=[Modality.CAMERAS, Modality.LIDAR_PC],
           frame_selector=CurrentSelector(location="features"),
       ),
       # 0.5 seconds in the past (no built-in selector for this)
       # Use ClosestTimestampSelector with negative delta_t
       FrameLoader(
           frame_name="past_0.5s",
           required_modalities=[Modality.CAMERAS],
           frame_selector=ClosestTimestampSelector(
               location="features",
               delta_t=-0.5,
           ),
       ),
   ]
   
   # Or use config-based approach
   config = {
       "dataset": {
           "features": [
               {
                   "frame_name": "current",
                   "modalities": ["cameras"],
                   "selector": {"name": "current"}
               },
               {
                   "frame_name": "past_1s",
                   "modalities": ["cameras"],
                   "selector": {
                       "name": "closest_timestamp",
                       "parameters": {"delta_t": -1.0}
                   }
               },
           ]
       }
   }

Custom Filtering
~~~~~~~~~~~~~~~~

Advanced index filtering:

.. code-block:: python

   import pandas as pd
   
   index = pd.read_parquet("/data/processed/index.parquet")
   
   # Complex filter: daytime urban scenes with good weather
   filtered = index[
       (index['time_of_day'] == 'day') &
       (index['location_type'] == 'urban') &
       (index['weather'].isin(['clear', 'cloudy'])) &
       (index['num_detections'] > 5)  # Scenes with traffic
   ]
   
   # Sample subset for faster iteration
   sampled = filtered.sample(n=10000, random_state=42)
   
   dataset = UnifiedE2EDataset(sampled, ...)

Preprocessing Datasets
----------------------

Waymo E2E Example
~~~~~~~~~~~~~~~~~

Preprocess Waymo End-to-End dataset:

.. code-block:: bash

   python examples/dataset_preprocessing.py \
       --e2e_dataset_path /raw/waymo/e2e/training \
       --split training \
       --processed_data_path /processed/waymo_e2e \
       --num_workers 8

This creates:

.. code-block:: text

   /processed/waymo_e2e/
   ├── index.parquet          # Frame metadata
   ├── features/              # Preprocessed features
   │   ├── frame_0001.npz
   │   ├── frame_0002.npz
   │   └── ...
   └── labels/                # Preprocessed labels
       ├── frame_0001.npz
       ├── frame_0002.npz
       └── ...

Custom Dataset Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the `Adding New Datasets Guide <https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_ for detailed instructions on adding your own dataset.

Supported Datasets
------------------

Current Status
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Dataset
     - Status
     - Notes
   * - Waymo Open E2E
     - ✅ Supported
     - Full multimodal support
   * - Waymo Open Perception
     - 🟡 In Progress
     - Core functionality working
   * - Argoverse 2
     - 📋 Planned
     - Coming soon
   * - nuScenes
     - 📋 Planned
     - Coming soon
   * - KITTI
     - 📋 Planned
     - Coming soon

Want to add a dataset? Check out our `contribution guide <https://github.com/stepankonev/StandardE2E/blob/main/CONTRIBUTING.md>`_!

Testing & Development
---------------------

Running Tests
~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

   pytest

Run specific test suites:

.. code-block:: bash

   # Dataset processors only
   pytest tests/dataset_processors -v
   
   # Frame data tests
   pytest tests/test_frame_data.py -v
   
   # With coverage
   pytest --cov=standard_e2e --cov-report=html

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build docs locally:

.. code-block:: bash

   cd docs
   make html
   
   # View in browser
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Clean build:

.. code-block:: bash

   make clean
   make html

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'standard_e2e'**

Solution: Install in editable mode:

.. code-block:: bash

   pip install -e .

**CUDA out of memory**

Solutions:

- Reduce batch size
- Reduce number of DataLoader workers
- Enable gradient checkpointing
- Use mixed precision training

.. code-block:: python

   dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

**Slow data loading**

Solutions:

- Increase num_workers
- Enable pin_memory
- Preload index into memory
- Use SSD for processed data

.. code-block:: python

   dataloader = DataLoader(
       dataset,
       num_workers=8,
       pin_memory=True,
       persistent_workers=True
   )

**Missing modality data**

Use modality defaults:

.. code-block:: python

   from standard_e2e.dataset_utils.modality_defaults import (
       IntentDefaults,
       PreferredTrajectoryDefaults,
   )
   from standard_e2e.enums import Modality
   
   dataset = UnifiedE2EDataset(
       ...,
       modality_defaults={
           Modality.INTENT: IntentDefaults(),
           Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults(),
       }
   )

Getting Help
~~~~~~~~~~~~

- 📖 Check the :doc:`quickstart` and :doc:`overview`
- 🔍 Search `GitHub Issues <https://github.com/stepankonev/StandardE2E/issues>`_
- 💬 Ask in `GitHub Discussions <https://github.com/stepankonev/StandardE2E/discussions>`_
- 📧 Contact maintainers

Next Steps
----------

- 📓 Try the `interactive notebooks <https://github.com/stepankonev/StandardE2E/tree/main/notebooks>`_
- 🔍 Explore the :doc:`reference/api` for detailed API docs
- 🚀 Check out the `examples <https://github.com/stepankonev/StandardE2E/tree/main/examples>`_
- 🤝 Contribute to the project!
