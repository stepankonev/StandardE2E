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

Built-ins
---------

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.adapters.PanoImageAdapter
   standard_e2e.caching.adapters.IntentIdentityAdapter
   standard_e2e.caching.adapters.FutureStatesIdentityAdapter
   standard_e2e.caching.adapters.PastStatesIdentityAdapter
   standard_e2e.caching.adapters.PreferenceTrajectoryAdapter
   standard_e2e.caching.adapters.Detections3DIdentityAdapter
