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
     - ✓ (8 ring cameras)
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
   * - `Argoverse 2 Lidar <https://www.argoverse.org/av2.html#lidar-link>`__
     - —
     - ✓ (merged sweep, ego frame)
     - ✓
     - —
     - —
     - —
   * - `NAVSIM <https://github.com/autonomousvision/navsim>`__ (OpenScene-v1.1)
     - ✓ (8 cameras: front/left×3/right×3/rear)
     - ✓ (merged sweep, ego frame)
     - ✓ (via nuPlan ``map.gpkg`` → unified taxonomy; lane boundaries
       carry no paint info, since nuPlan doesn't store paint type)
     - ✓
     - ✓ (4-class one-hot → :class:`~standard_e2e.enums.Intent`)
     - —
   * - `WayveScenes101 <https://wayve.ai/science/wayvescenes101>`__
     - ✓ (5 fisheye: forward + side arc)
     - ✓ (COLMAP SfM, ego frame) [#wayve_lidar]_
     - —
     - —
     - —
     - —
   * - `comma2k19 <https://github.com/commaai/comma2k19>`__
     - ✓ (1 forward: comma EON, 1164×874 pinhole) [#comma2k19]_
     - —
     - —
     - —
     - —
     - —
   * - `TruckDrive <https://torc-ai.github.io/TruckDrive/>`__ (heavy-truck)
     - ✓ (11-camera 8 MP rig) [#truckdrive]_
     - ✓ (Aeva FMCW joint cloud, ego frame)
     - —
     - ✓ (ego frame)
     - —
     - —
   * - `nuScenes <https://www.nuscenes.org/nuscenes>`__
     - ✓ (6 surround cameras) [#nuscenes]_
     - ✓ (LIDAR_TOP, ego frame)
     - ✓ (map-expansion → unified taxonomy)
     - ✓ (ego frame)
     - —
     - —

All datasets also emit the ego **past/future trajectory** (from each
dataset's poses, via the segment-context aggregator) regardless of the
columns above.

.. note::

   **comma2k19 is high-volume** — 20 Hz × ~2000 one-minute segments ≈ 2.4 M
   frames (~2 TB at the native 1164×874 resolution). Two converter knobs bound
   the output size and processing time: ``--frame_stride N`` keeps **every
   N-th frame** (``1`` = full 20 Hz; e.g. ``--frame_stride 4`` ≈ 5 Hz), and
   the ``cameras_identity_adapter``'s ``max_size`` param **downscales** each
   frame so its longest side is at most that many px (intrinsics scaled to
   match).

.. [#wayve_lidar] WayveScenes101 ships **no sensor lidar**. Its ``lidar_pc``
   is populated from the per-scene **COLMAP SfM** point cloud: filtered
   (reprojection error ≤ 6, track length ≥ 2), converted OpenCV→FLU, then
   transformed into each frame's ego (FLU, x-forward/y-left/z-up) frame with
   the *world→ego* pose and range-clipped (50 m) so it flows through the
   standard lidar adapters. It is photogrammetric (sparse, up-to-scale), not
   a sensor measurement. The ego, cameras and lidar share one FLU frame, so
   a frame's cloud lifted by ``aux_data["pose_matrix"]`` reproduces the
   source SfM cloud exactly.

.. [#comma2k19] comma2k19 ships a **single forward-facing** 20 Hz camera
   (comma EON, 1164×874, treated as a pinhole; identity extrinsics, since the
   dataset pose *is* the camera pose) plus a fused GNSS/IMU ego pose and CAN
   telemetry — no lidar, HD map, 3D boxes, or driving command. The ego pose is
   derived from the ECEF ``global_pose`` into a per-segment local FLU frame
   (x-forward/y-left/z-up), so ``global_position`` X/Y/Z/heading are
   segment-relative; ``global_position`` additionally carries the ego
   **speed** (:attr:`~standard_e2e.enums.TrajectoryComponent.SPEED`) from the
   ECEF velocity. Segments must be extracted from the distributed
   ``Chunk_*.zip`` archives first (as with WayveScenes101); each ``video.hevc``
   is then decoded forward-only, since HEVC random seek is unreliable. Native
   rate is 20 Hz — use ``--frame_stride`` to subsample.

.. [#truckdrive] TruckDrive (Torc Robotics / Princeton, CVPR 2026) is a
   long-range highway **heavy-truck** dataset. Its 8 MP surround rig has
   **11 cameras** — more than the eight canonical
   :class:`~standard_e2e.enums.CameraDirection` slots — so each camera is
   mapped to the canonical member matching its facing wherever one fits, with
   dedicated members added only for the extra views the eight can't name: the
   forward telephoto pair (``FRONT_LEFT_NARROW`` / ``FRONT_RIGHT_NARROW``) and
   the rear-facing side pair (``SIDE_LEFT_BACK`` / ``SIDE_RIGHT_BACK``).
   ``lidar_pc`` is the seven-sensor **Aeva FMCW** joint cloud (xyz kept,
   transformed into the ego ``velodyne`` frame); ``detections_3d`` are the
   tracked 3D boxes in the ego frame, with the ego vehicle's own cab/trailer
   and ``DontCare`` groups excluded per the paper taxonomy. The ego pose
   (PPK + LiDAR-SLAM) drives the past/future trajectory. Short-range Ouster
   lidar, 4D radar, accumulated GT depth and lane lines have no StandardE2E
   target yet and are not ingested. The dataset ships as per-scene,
   per-modality zips and **must be extracted first**
   (``scripts/extract_truckdrive.sh``, or
   ``scripts/prepare_dataset_truckdrive.sh`` to extract and preprocess in one
   step); frames are matched across sensors by their synchronization key.

.. [#nuscenes] nuScenes (Motional, CVPR 2020) is the de-facto surround-view
   E2E / BEV benchmark: 1000 ~20 s scenes, a 6-camera surround rig (1600×900),
   a 32-beam ``LIDAR_TOP`` and densely annotated 3D boxes at 2 Hz keyframes
   (one frame = one keyframe sample; one segment = one scene). The six ``CAM_*``
   channels map onto the canonical
   :class:`~standard_e2e.enums.CameraDirection` members; ``lidar_pc`` is the
   ``LIDAR_TOP`` cloud (xyz, in the ego frame); ``detections_3d`` are the
   ``sample_annotation`` boxes transformed from the global frame into the ego
   frame, each ``category_name`` folded into the coarse
   :class:`~standard_e2e.enums.DetectionType`. The vector **map-expansion**
   (lane centers from the arcline paths, lane/road dividers, crossings,
   walkways, stop lines, drivable area, intersections) is translated to the
   unified :class:`~standard_e2e.enums.MapElementType` in the ego frame and
   rasterised by ``HDMapBEVAdapter`` -- only when the separate
   ``nuScenes-map-expansion-v1.3`` pack is unzipped into ``<dataroot>/maps/``
   (else the HD map is skipped). ``--split`` is an official nuScenes label that
   also selects the metadata version (``mini_train`` / ``mini_val`` →
   ``v1.0-mini``, ``train`` / ``val`` → ``v1.0-trainval``, ``test`` →
   ``v1.0-test``); the test split ships no annotations. nuScenes is read
   **directly from the JSON tables** -- the ``nuscenes-devkit`` is not a runtime
   dependency (it pins ``numpy<2``), so the split scene-lists and the
   lane-arcline discretization are vendored from it (Apache-2.0). The 5 radars
   have no StandardE2E target yet. A partially-downloaded trainval converts
   cleanly: scenes whose sensor blob is not yet on disk are skipped. The release
   ships as ``.tgz`` archives and must be extracted first
   (``scripts/extract_nuscenes.sh``, or ``scripts/prepare_dataset_nuscenes.sh``
   to extract and preprocess in one step).

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
