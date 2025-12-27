Caching and Processing
======================

Core Processing
---------------

Base Classes
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.source_dataset_processor.SourceDatasetProcessor
   standard_e2e.caching.source_dataset_converter.SourceDatasetConverter


.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.process_source_dataset.process_source_dataset

Dataset Processors
------------------

Waymo End-to-End
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.src_datasets.waymo_e2e.WaymoE2EDatasetProcessor
   standard_e2e.caching.src_datasets.waymo_e2e.WaymoE2EDatasetConverter

Waymo Perception
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.src_datasets.waymo_perception.WaymoPerceptionDatasetProcessor
   standard_e2e.caching.src_datasets.waymo_perception.WaymoPerceptionDatasetConverter

Adapters
--------

Base Adapter
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.AbstractAdapter

Built-in Adapters
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.PanoImageAdapter
   standard_e2e.caching.adapters.IntentIdentityAdapter
   standard_e2e.caching.adapters.FutureStatesIdentityAdapter
   standard_e2e.caching.adapters.PastStatesIdentityAdapter
   standard_e2e.caching.adapters.PreferenceTrajectoryAdapter
   standard_e2e.caching.adapters.Detections3DIdentityAdapter

Segment Context Aggregators
----------------------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.segment_context.SegmentContextAggregator
   standard_e2e.caching.segment_context.FuturePastStatesFromMatricesAggregator
   standard_e2e.caching.segment_context.FutureDetectionsAggregator
