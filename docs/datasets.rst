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
   * - `nuScenes <https://www.nuscenes.org/nuscenes>`__
     - ✓ (6 surround cameras) [#nuscenes]_
     - ✓ (LIDAR_TOP, ego frame)
     - ✓ (map-expansion → unified taxonomy)
     - ✓ (ego frame)
     - —
     - —
   * - `KITScenes Multimodal <https://www.kitscenes.com/multimodal/>`__
     - ✓ (6 surround ring cameras) [#kitscenes]_
     - ✓ (lidar_top, ego frame)
     - ✓ (Lanelet2 → unified taxonomy)
     - —
     - —
     - —
   * - `KITScenes LongTail <https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail>`__
     - ✓ (6 surround ring cameras) [#kitscenes_longtail]_
     - —
     - —
     - —
     - ✓ (driving instruction)
     - ✓ (counterfactual futures)
   * - `NATIX Multi-Camera <https://huggingface.co/datasets/natix-network-org/natix-multi-camera-driving-dataset>`__ (crowd-sourced dashcam)
     - ✓ (4- or 6-camera Tesla rig) [#natix_multicam]_
     - —
     - —
     - —
     - —
     - —
   * - `NATIX Edge Case <https://huggingface.co/datasets/natix-network-org/natix-edge-case-driving-dataset>`__ (curated edge cases)
     - ✓ (4- or 6-camera Tesla rig) [#natix_edgecase]_
     - —
     - —
     - —
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
   ambiguous boxes → ``UNKNOWN``; ``DontCare`` dropped). Box yaw is VoD's KITTI
   rotation about the LiDAR **-Z** axis (camera-x zero-reference), so the ego
   heading is ``-(rotation + pi/2)``. One
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

.. [#kitscenes] KITScenes Multimodal (KIT / MRT, arXiv:2606.02956) -- the
   StandardE2E dataset key is ``kitscenes_multimodal``, distinct from its
   long-tail sibling **KITScenes LongTail** (key ``kitscenes_longtail``,
   [#kitscenes_longtail]_) -- is a large
   **European urban** dataset (~1000 scenes at 10 Hz) whose headline annotation
   is a dense, georeferenced **Lanelet2 HD map**. StandardE2E ingests its six
   surround **ring** cameras (``camera_ring_*`` → the canonical
   :class:`~standard_e2e.enums.CameraDirection` surround members; pinhole ``K``,
   ``T_ego_from_camera`` extrinsics from the calib's ``T_to_reference``), the
   128-beam **lidar_top** ``lidar_pc`` (xyz de-discretized from its int32 storage
   with the invalid-return sentinels dropped, in the ego ``base_frame`` whose
   origin is ``lidar_top``), and the **HD map**: each scene's Lanelet2 OSM
   (``maps/map.osm``) is parsed directly and translated to the unified
   :class:`~standard_e2e.enums.MapElementType` taxonomy -- lanelet
   road/emergency/bicycle → ``LANE_CENTER`` (centerline from the left/right bound
   midpoints), ``crosswalk`` → ``CROSSWALK``,
   ``line_thin`` / ``line_thick`` / ``bike_marking`` → ``LANE_BOUNDARY``,
   ``curbstone`` / ``road_border`` → ``ROAD_EDGE``, ``stop_line`` →
   ``STOP_LINE``, ``traffic_light*`` → ``TRAFFIC_LIGHT`` -- then cropped to an
   ego-centric ROI per frame and rasterised by ``HDMapBEVAdapter``. The
   georeferencing needs no GNSS reconciliation: ``poses.txt`` is already in the
   Lanelet2 map-local frame (UTM zone 32N minus the ``origin.json`` anchor),
   verified by the ego trajectory lying on the map's node cloud; the map is
   projected with ``pyproj`` (no ``lanelet2`` runtime dependency -- it pins
   ``numpy<2``, like the nuScenes devkit). One frame = one synchronized 10 Hz
   snapshot; one segment = one scene (a UUID directory); the reference timeline
   gives per-frame timestamps and ``poses.txt`` the per-frame ego pose, which
   drives the past/future trajectory. KITScenes ships **no 3D object boxes** (the
   HD map is its annotation product), so ``detections_3d`` is always empty. Not
   ingested: the three "base" cameras (hi-res front-center + rectified stereo
   pair), the other six LiDARs, the three imaging radars and the GNSS/INS
   streams. Splits follow the official geo-disjoint folders
   (``train`` / ``val`` / ``test`` / ``test_e2e`` / ``overlap_train_val``); a
   flat layout (e.g. the single-scene sample) processes every scene with
   ``--split`` as the output label. The release ships as per-split tarballs on
   Hugging Face (``KIT-MRT/KITScenes-Multimodal``) and **must be downloaded and
   extracted first** (``scripts/extract_kitscenes.sh``, or
   ``scripts/prepare_dataset_kitscenes_multimodal.sh`` to extract and preprocess
   in one step).

.. [#kitscenes_longtail] KITScenes LongTail (KIT / MRT, arXiv:2603.23607) is the
   **long-tail / reasoning-traces** sibling of KITScenes Multimodal (key
   ``kitscenes_longtail``) -- ~1000 9 s scenarios deliberately filtered for
   **rare events** (adverse weather, construction, night, road closures,
   overtaking / lane changes). Unlike Multimodal's per-scene sensor directories
   it ships as Hugging Face ``datasets`` **parquet**, one **scenario per row**,
   with a different modality set. Each scenario is a 360° multi-view video, so it
   is **unrolled into one frame per timestep** (5 Hz; one frame = one timestep,
   one segment = one scenario, ~21 frames over the 4 s observation window).
   StandardE2E ingests, per frame, the six surround **ring** cameras at that
   timestep (mapped to the canonical
   :class:`~standard_e2e.enums.CameraDirection` surround members with the
   fixed-rig pinhole ``K`` and ``T_ego_from_camera`` from the README's
   ``(K, R, t)``, where ``R`` is ego→camera so the extrinsics are
   ``inv(se3(R, t))``), the high-level **driving_instruction** folded into the
   coarse :class:`~standard_e2e.enums.Intent` (``left`` → ``GO_LEFT``, ``right``
   → ``GO_RIGHT``, ``straight`` → ``GO_STRAIGHT``, else ``UNKNOWN``; the raw
   instruction kept in ``aux_data``), and **past_states** -- the ego history up
   to that timestep, re-expressed **ego-relative to the frame** (ego at the
   origin, facing +x along its path tangent), metres (x forward, y left). The
   **prediction targets** -- the 5 s **expert** future (``future_states``, 25
   pts) and the **counterfactual** futures (``wrong_speed`` /
   ``neglect_instruction`` / ``off_road`` / ``crash``) as
   **``preference_trajectory``** (only the ones present) -- are attached to the
   ``t=0`` (last) frame only (marked ``is_prediction_frame`` in the index). It
   ships **no lidar, HD map or 3D boxes**.
   The multilingual **reasoning traces** (English / Spanish / Chinese) are
   surfaced (English) into ``aux_data``. Note: the released **test** split
   withholds the future trajectories (eval ground truth -- a ``[[-100, -100]]``
   sentinel); only **train** / **train_raw** carry real expert + counterfactual
   futures. The **_raw** splits are native-resolution frames; the non-``raw``
   splits are the processed frames the vendored intrinsics match. The release
   ships as gated Hugging Face parquet (``KIT-MRT/KITScenes-LongTail``) and
   **must be downloaded first**
   (``scripts/prepare_dataset_kitscenes_longtail.sh``).

.. [#natix_multicam] NATIX Multi-Camera (key ``natix_multicam``; its curated
   edge-case sibling is supported as ``natix_edgecase`` -- see
   [#natix_edgecase]_ -- and NATIX has also announced a telemetry-included
   release) is a **crowd-sourced** real-world driving dataset: 100 h of
   **Tesla-dashcam** surround footage from everyday, non-expert drivers in
   Switzerland and the United States -- ~1-minute mp4 clips at ~36 fps plus
   per-clip **consumer-grade GPS** metadata (1-10 Hz), per-trip camera
   calibration and trip-level metadata. One **frame = one front-camera GPS
   fix** (the video between fixes carries no pose and is not emitted;
   ``--frame_stride`` subsamples further); one **segment = one trip piece**
   (continuous minutes), its clips concatenated chronologically with the
   wall-clock CSV timestamps localized via ``trip_insight.json``'s IANA
   timezone. Cameras map to the canonical
   :class:`~standard_e2e.enums.CameraDirection` by **facing**, verified from
   the calibration rotations: in 4-camera trips ``LEFT`` / ``RIGHT`` *are*
   the backward-facing repeaters → ``REAR_LEFT`` / ``REAR_RIGHT`` (not
   ``SIDE_*``); the 6-camera pillar pair faces forward-side → ``FRONT_LEFT``
   / ``FRONT_RIGHT``. Pinhole ``K`` + Brown-Conrady distortion as shipped
   (the front camera can be natively 2896×1876; its ``K`` already matches),
   extrinsics ``T_ego_from_camera`` in the **optical** frame (converted from
   the ``ground_nominal`` body-FLU rotations, cm → m). Per-camera streams
   differ in frame rate and duration, so each camera is matched to the fix
   **by timestamp**; a camera with no usable match is absent from that
   frame's ``cameras``. The ego pose is GPS-derived (per-segment
   azimuthal-equidistant local east/north frame anchored at the first fix,
   z = 0, yaw from ``heading_deg`` with unreliable headings held through
   stops) and drives the past/future trajectories; ``global_position`` also
   carries the fix **speed**. ``extra_index_data`` carries trip / country /
   region / camera_count. No lidar, HD map, 3D boxes or driving command; no
   canonical split (``--split`` is a passthrough label). Source-data quality
   bounds (documented, not corrected): GPS accuracy is tens of metres,
   video/GPS sync error 0-1 s (extremes to 3 s), calibrations are nominal
   per vehicle model. **No extraction step** -- the gated Hugging Face repo
   carries a ~20-minute ``dataset-sample/``; the full ~1.28 TB is pulled
   from a Cloudflare R2 bucket with credentials granted on access approval
   (``scripts/prepare_dataset_natix_multicam.sh``).

.. [#natix_edgecase] NATIX Edge Case (key ``natix_edgecase``) is the
   **curated edge-case sibling** of NATIX Multi-Camera: rare, challenging
   real-world scenarios -- construction zones, adverse weather, road-surface
   deterioration, illegal maneuvers, obstructions -- crowd-sourced from the
   same decentralized Tesla-dashcam network. The first public release is
   **20 minutes / 86 mp4 clips** across six US states (4- and 6-camera
   rigs) with **21 VLM-annotated events**. Footage, GPS metadata,
   calibration and folder layout are **identical to** ``natix_multicam``
   (same frame/segment definitions, camera mapping, GPS ego pose and
   data-quality bounds -- everything in [#natix_multicam]_ applies; the
   processor is a subclass). The release adds the **edge-case annotations**
   (``data/edge-case.json``): per annotated clip, one or more events, each
   with a ``label`` (may be empty), a ``[start_sec, end_sec]`` window in
   **video seconds from clip start**, and a structured **VLM**
   ``ai_analysis`` (event classification, visual evidence, context, agentic
   validation, final detected event). Every emitted frame carries the events
   covering its timestamp -- the full payload in
   ``aux_data["edge_case_events"]``, and ``edge_case_count`` /
   ``edge_case`` (summaries; a missing label falls back to the VLM ``EVENT
   CLASSIFICATION``) in the index -- so in-event frames are filterable via
   ``extra_edge_case_count > 0`` without touching the npz. Windows map onto
   the frame timeline through each clip's front CSV (the epoch of video
   frame 1, extrapolated when the CSV's leading rows are missing). The
   annotations are **VLM-generated best-effort** (integer-second windows,
   model-written descriptions) -- guidance, not ground truth. One further
   observed GPS degradation (passed through as-shipped): a trip's fixes can
   **freeze** (lat/lon stuck, speed 0) for tens of seconds of moving
   footage, leaving the pose-derived past/future trajectories degenerate
   over that stretch even though the cameras are fine. **No
   extraction step** -- the gated Hugging Face repo
   (``natix-network-org/natix-edge-case-driving-dataset``, ~3.3 GB) carries
   everything under ``data/``
   (``scripts/prepare_dataset_natix_edgecase.sh``).

How datasets are added
----------------------

See `Adding New Datasets Guide
<https://github.com/stepankonev/StandardE2E/blob/main/standard_e2e/caching/src_datasets/adding_new_dataset.md>`_
for the full processor → adapter → aggregator pipeline a new dataset has
to plug into.
