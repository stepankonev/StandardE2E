Supported Datasets
==================

Each row below lists the modalities the dataset processor currently emits
into the unified format. ``✓`` means the modality is wired end-to-end
through preprocessing and the on-disk ``TransformedFrameData`` schema;
``—`` means the dataset doesn't ship that modality (and it's surfaced as
the corresponding default value via
:class:`~standard_e2e.dataset_utils.modality_defaults.ModalityDefaults`).

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 10 10 12 14

   * - Dataset
     - Cameras
     - LiDAR
     - HD map (BEV)
     - 3D detections
     - Driving command
     - Preference trajectory
   * - `Waymo Open E2E <https://waymo.com/open/data/e2e/>`__
     - ✓ (8 ring cameras, panoramic adapter)
     - —
     - —
     - —
     - ✓
     - ✓
   * - `Waymo Open Perception <https://waymo.com/open/data/perception/>`__
     - ✓ (5 cameras)
     - ✓ (top + side, ego frame)
     - ✓
     - ✓
     - —
     - —
   * - `Argoverse 2 Sensor <https://www.argoverse.org/av2.html#sensor-link>`__
     - ✓ (7 ring cameras)
     - ✓ (merged sweep, ego frame)
     - ✓
     - ✓
     - —
     - —

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
