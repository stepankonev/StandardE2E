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
   * - `View-of-Delft <https://intelligent-vehicles.org/datasets/view-of-delft/>`__ (urban radar)
     - ✓ (1 front: 1936×1216 pinhole) [#vod]_
     - ✓ (Velodyne HDL-64, ego frame)
     - —
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

.. [#vod] View-of-Delft (TU Delft, IEEE RA-L 2022) is a compact **urban**
   dataset whose distinctive sensor is a **3+1D radar**. StandardE2E ingests its
   single **front camera** (:class:`~standard_e2e.enums.CameraDirection.FRONT`,
   1936×1216 pinhole; intrinsics from the calib ``P2``, extrinsics from
   ``inv(Tr_velo_to_cam)``), the 64-layer **Velodyne** ``lidar_pc`` (xyz in the
   ego ``velodyne`` frame, per-point reflectance dropped) and KITTI
   ``detections_3d`` mapped from camera coordinates into the ego frame -- each of
   VoD's 13 classes folded into the coarse
   :class:`~standard_e2e.enums.DetectionType` (the two-wheeler family
   ``bicycle`` / ``Cyclist`` / ``rider`` / ``moped_scooter`` / ``motor`` →
   ``BICYCLE``; ``Car`` / ``truck`` / ``vehicle_other`` → ``VEHICLE``; static or
   ambiguous boxes → ``UNKNOWN``; ``DontCare`` dropped). Box yaw is VoD's
   rotation about the LiDAR **-Z** axis, so the ego heading is its negation. One
   frame = one keyframe; one segment = one recording scene (``delft_*``), grouped
   via the official scene table so the per-segment past/future ego trajectory
   (from the per-frame ``mapToCamera`` pose) never spans two recordings. The
   **3+1D radar** (``radar`` / ``radar_3frames`` / ``radar_5frames``) has no
   StandardE2E modality yet and is not ingested; the detection release ships no
   per-frame timestamps, so they are **synthesised** at the 10 Hz LiDAR-lead
   rate; the **test** split has sensor data but no labels. The dataset ships as
   zips and **must be extracted first** (``scripts/extract_vod.sh`` -- lidar tree
   only, with an optional track-id overlay -- or
   ``scripts/prepare_dataset_vod.sh`` to extract and preprocess in one step).

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
