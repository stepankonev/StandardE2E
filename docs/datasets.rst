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

All datasets also emit the ego **past/future trajectory** (from each
dataset's poses, via the segment-context aggregator) regardless of the
columns above.

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
   ECEF velocity. Segments are read from extracted directories or the
   distributed ``Chunk_*.zip`` archives directly; each ``video.hevc`` is
   decoded forward-only, since HEVC random seek is unreliable. Native rate is
   20 Hz — use ``--frame_stride`` to subsample.

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
