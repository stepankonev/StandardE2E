Enums
=====

Modality
--------

The **output** side of the adapter contract: the keys under which adapters
write their results into
:class:`~standard_e2e.data_structures.frame_data.TransformedFrameData`, and
the modality whitelist the loader reads back at training time. Each adapter's
``transform`` returns ``{Modality.X: payload}`` — e.g.
:class:`~standard_e2e.caching.adapters.LidarAdapter` produces
``Modality.LIDAR_PC`` and
:class:`~standard_e2e.caching.adapters.HDMapBEVAdapter` produces
``Modality.HD_MAP_BEV``.

Contrast with :class:`~standard_e2e.enums.StandardFrameDataField`, the
**input** side (the raw ``StandardFrameData`` fields adapters consume). The
relationship is **many-to-one**: the single input field ``LIDAR`` feeds both
the ``LIDAR_PC`` and ``LIDAR_BEV`` output modalities, and ``HD_MAP`` feeds
``HD_MAP`` and ``HD_MAP_BEV`` — which is exactly why the two are separate
enums rather than one.

.. autoclass:: standard_e2e.enums.Modality
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

StandardFrameDataField
----------------------

Field names of :class:`~standard_e2e.data_structures.frame_data.StandardFrameData`
— the **input** side of the adapter contract: what an adapter declares it
reads via :attr:`~standard_e2e.caching.adapters.AbstractAdapter.consumes_attrs`,
and what a processor gates its (often expensive) modality builds on via
:meth:`~standard_e2e.caching.source_dataset_processor.SourceDatasetProcessor.needs_attr`.
Distinct from :class:`~standard_e2e.enums.Modality`, the **output** side: one
input field (e.g. ``LIDAR``) can feed several output modalities (``LIDAR_PC``,
``LIDAR_BEV``).

.. autoclass:: standard_e2e.enums.StandardFrameDataField
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

Intent
------

Enumeration of driver intent values.

.. autoclass:: standard_e2e.enums.Intent
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

CameraDirection
---------------

Enumeration of camera mounting directions.

.. autoclass:: standard_e2e.enums.CameraDirection
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

TrajectoryComponent
-------------------

Enumeration of trajectory data components.

.. autoclass:: standard_e2e.enums.TrajectoryComponent
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value

DetectionType
-------------

Enumeration of object detection types.

.. autoclass:: standard_e2e.enums.DetectionType
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: name, value
