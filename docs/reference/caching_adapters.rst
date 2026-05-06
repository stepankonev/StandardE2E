Caching Adapters
================

Base adapter interface and built-in implementations for transforming
``StandardFrameData`` into modality-specific payloads.

Base
----

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.AbstractAdapter

Camera Adapters
---------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.PanoImageAdapter
   standard_e2e.caching.adapters.CamerasIdentityAdapter

Intent Adapters
---------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.IntentIdentityAdapter

Trajectory Adapters
-------------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.FutureStatesIdentityAdapter
   standard_e2e.caching.adapters.PastStatesIdentityAdapter
   standard_e2e.caching.adapters.PreferenceTrajectoryAdapter

Detection Adapters
------------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.Detections3DIdentityAdapter

LiDAR Adapters
--------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.LidarAdapter
   standard_e2e.caching.adapters.LidarBEVAdapter

HD Map Adapters
---------------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.HDMapBEVAdapter
