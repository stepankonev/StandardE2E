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
   * - `Waymo Open E2E <https://github.com/stepankonev/StandardE2E/blob/main/datasets/waymo_e2e/README.md>`__
     - ✓ (8 ring cameras)
     - —
     - —
     - —
     - ✓
     - ✓
   * - `Waymo Open Perception <https://github.com/stepankonev/StandardE2E/blob/main/datasets/waymo_perception/README.md>`__
     - ✓ (5 cameras)
     - ✓ (top + side, ego frame)
     - ✓
     - ✓
     - —
     - —
   * - `Argoverse 2 Sensor <https://github.com/stepankonev/StandardE2E/blob/main/datasets/av2_sensor/README.md>`__
     - ✓ (7 ring cameras)
     - ✓ (merged sweep, ego frame)
     - ✓
     - ✓
     - —
     - —
   * - `Argoverse 2 Lidar <https://github.com/stepankonev/StandardE2E/blob/main/datasets/av2_lidar/README.md>`__
     - —
     - ✓ (merged sweep, ego frame)
     - ✓
     - —
     - —
     - —
   * - `NAVSIM <https://github.com/stepankonev/StandardE2E/blob/main/datasets/navsim/README.md>`__ (OpenScene-v1.1)
     - ✓ (8 cameras: front/left×3/right×3/rear)
     - ✓ (merged sweep, ego frame)
     - ✓ (via nuPlan ``map.gpkg`` → unified taxonomy; lane boundaries
       carry no paint info, since nuPlan doesn't store paint type)
     - ✓
     - ✓ (4-class one-hot → :class:`~standard_e2e.enums.Intent`)
     - —
   * - `WayveScenes101 <https://github.com/stepankonev/StandardE2E/blob/main/datasets/wayve_scenes/README.md>`__
     - ✓ (5 fisheye: forward + side arc)
     - ✓ (COLMAP SfM, ego frame)
     - —
     - —
     - —
     - —
   * - `comma2k19 <https://github.com/stepankonev/StandardE2E/blob/main/datasets/comma2k19/README.md>`__
     - ✓ (1 forward: comma EON, 1164×874 pinhole)
     - —
     - —
     - —
     - —
     - —
   * - `TruckDrive <https://github.com/stepankonev/StandardE2E/blob/main/datasets/truckdrive/README.md>`__ (heavy-truck)
     - ✓ (11-camera 8 MP rig)
     - ✓ (Aeva FMCW joint cloud, ego frame)
     - —
     - ✓ (ego frame)
     - —
     - —
   * - `View-of-Delft <https://github.com/stepankonev/StandardE2E/blob/main/datasets/vod/README.md>`__ (urban radar)
     - ✓ (1 front: 1936×1216 pinhole)
     - ✓ (Velodyne HDL-64, ego frame)
     - —
     - ✓ (ego frame)
     - —
     - —
   * - `nuScenes <https://github.com/stepankonev/StandardE2E/blob/main/datasets/nuscenes/README.md>`__
     - ✓ (6 surround cameras)
     - ✓ (LIDAR_TOP, ego frame)
     - ✓ (map-expansion → unified taxonomy)
     - ✓ (ego frame)
     - —
     - —
   * - `KITScenes Multimodal <https://github.com/stepankonev/StandardE2E/blob/main/datasets/kitscenes_multimodal/README.md>`__
     - ✓ (6 surround ring cameras)
     - ✓ (lidar_top, ego frame)
     - ✓ (Lanelet2 → unified taxonomy)
     - —
     - —
     - —
   * - `KITScenes LongTail <https://github.com/stepankonev/StandardE2E/blob/main/datasets/kitscenes_longtail/README.md>`__
     - ✓ (6 surround ring cameras)
     - —
     - —
     - —
     - ✓ (driving instruction)
     - ✓ (counterfactual futures)
   * - `NATIX Multi-Camera <https://github.com/stepankonev/StandardE2E/blob/main/datasets/natix_multicam/README.md>`__ (crowd-sourced dashcam)
     - ✓ (4- or 6-camera Tesla rig)
     - —
     - —
     - —
     - —
     - —
   * - `NATIX Edge Case <https://github.com/stepankonev/StandardE2E/blob/main/datasets/natix_edgecase/README.md>`__ (curated edge cases)
     - ✓ (4- or 6-camera Tesla rig)
     - —
     - —
     - —
     - —
     - —
   * - `MUHOS <https://github.com/stepankonev/StandardE2E/blob/main/datasets/muhos/README.md>`__ (Finnish roads incl. gravel)
     - ✓ (front fisheye; calibration estimated)
     - ✓
     - —
     - —
     - —
     - —

All datasets also emit the ego **past/future trajectory** (from each
dataset's poses, via the segment-context aggregator) regardless of the
columns above.

Each dataset's folder in the repository, ``datasets/<name>/``, describes what
StandardE2E extracts from it and its caveats, and holds its isolated environment
and preprocessing pipeline (see :doc:`quickstart`). The dataset names above link
to those descriptions.

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
