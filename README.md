<div align="center">
  <img src="https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/standard_e2e_logo_contrast.png" alt="StandardE2E Logo" width="400"/>
  
  <p><em>A framework for unified end-to-end autonomous driving datasets processing</em></p>

  ![Python versions](https://img.shields.io/badge/Python-3.12-informational?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
  [![PyPI Downloads](https://static.pepy.tech/personalized-badge/standard-e2e?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/standard-e2e)
  [![Tests](https://github.com/stepankonev/StandardE2E/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/stepankonev/StandardE2E/actions/workflows/tests.yml)
  [![codecov](https://codecov.io/github/stepankonev/StandardE2E/graph/badge.svg?token=3MWJNB10OO)](https://codecov.io/github/stepankonev/StandardE2E)
  [![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
  [![mypy](https://img.shields.io/badge/mypy-checked-2A6DB2?logo=mypy&logoColor=white)](http://mypy-lang.org/)
  [![StandardE2E](https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/StandardE2E_gh_badge_dark.svg)](https://github.com/stepankonev/StandardE2E)
  
</div>

<div align="center">

  [![Docs](https://readthedocs.org/projects/standarde2e/badge/?version=latest)](https://standarde2e.readthedocs.io/en/latest/)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/vJnQNcQGQ8)

</div>

StandardE2E provides a consistent interface for preprocessing, loading, and training with multimodal data from various end-to-end autonomous driving datasets. It standardizes the complex process of working with different dataset formats, allowing researchers to focus on model development rather than data engineering.




![StandardE2E Architecture](https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/standard_e2e_scheme.png)

---

## 📖 Documentation

- Latest docs: https://standarde2e.readthedocs.io/en/latest/

## 📦 Installation

### Option 1: From PyPI (Recommended for Users)
```bash
pip install standard-e2e
```

### Option 2: Development with uv (recommended)
```bash
# Install uv: https://docs.astral.sh/uv/
git clone https://github.com/stepankonev/StandardE2E.git
cd StandardE2E
uv sync --all-extras   # installs deps and dev deps from uv.lock
uv run pytest tests/   # run tests
```

### Option 3: Manual development (pip/conda)
```bash
conda create -n standard_e2e python=3.12
conda activate standard_e2e
pip install -e ".[dev]"
```

## Plan for E2E Autonomous Driving Datasets Support


| Dataset | Cameras | Lidar | HD Map | Detections | Driving Command | Preference Trajectories |
|---------|---------|-------|--------|------------|-----------------|-------------------------|
| [Waymo End-to-end](https://waymo.com/open/data/e2e/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ❌ | ❌ | ❌ | ✅ | ✅ |
| [Waymo Perception](https://waymo.com/open/data/perception/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/semicircle-orange) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Navsim](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ✅ | ❌ |
| [WayveScenes101](https://wayve.ai/science/wayvescenes101) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/semicircle-orange) | ![](https://img.shields.io/badge/SfM-orange)¹ | ❌ | ❌ | ❌ | ❌ |
| [Argoverse 2 Sensor](https://www.argoverse.org/av2.html#sensor-link) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Lidar](https://www.argoverse.org/av2.html#lidar-link) ![](https://img.shields.io/badge/supported-darkgreen) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [comma2k19](https://github.com/commaai/comma2k19) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/front-darkred)² | ❌ | ❌ | ❌ | ❌ | ❌ |
| [TruckDrive](https://torc-ai.github.io/TruckDrive/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen)³ | ✅ | ❌ | ✅ | ❌ | ❌ |
| [View-of-Delft](https://intelligent-vehicles.org/datasets/view-of-delft/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/front-darkred)⁴ | ✅ | ❌ | ✅ | ❌ | ❌ |
| [nuScenes](https://www.nuscenes.org/nuscenes) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen)⁵ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [KITScenes Multimodal](https://www.kitscenes.com/multimodal/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen)⁶ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITScenes LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen)⁷ | ❌ | ❌ | ❌ | ✅ | ✅ |
| [NATIX Multi-Camera](https://huggingface.co/datasets/natix-network-org/natix-multi-camera-driving-dataset) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/ring_4--6_cam-orange)⁸ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [NATIX Edge Case](https://huggingface.co/datasets/natix-network-org/natix-edge-case-driving-dataset) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/ring_4--6_cam-orange)⁹ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Argoverse 2 Map Change](https://www.argoverse.org/av2.html#mapchange-link) ![](https://img.shields.io/badge/TBD-gray) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) + 2 x ![](https://img.shields.io/badge/side%20fisheye-blue) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [Waymo Motion Prediction](https://waymo.com/open/data/motion/) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Motion Forecasting [?]](https://www.argoverse.org/av2.html#forecasting-link) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |

¹ WayveScenes101 ships no sensor lidar; the `lidar_pc` modality is populated from the per-scene **COLMAP SfM** point cloud, transformed into each frame's ego frame and range-clipped so it flows through the lidar adapters. It is photogrammetric (sparse, up-to-scale), not a sensor measurement. Cameras are 5 fisheye views (forward + side arc); the ego past/future trajectory is derived from the COLMAP poses. Scenes must be extracted from the distributed `scene_<NNN>.zip` archives before processing.

² comma2k19 is a single forward-facing 20 Hz camera (comma EON, 1164×874, pinhole) with a fused GNSS/IMU ego pose; the ego past/future trajectory is derived from the global ECEF poses and `global_position` additionally carries the ego **speed**. It ships no lidar, HD map, 3D boxes, or driving command. Segments must be extracted from the distributed `Chunk_*.zip` archives before processing (as with WayveScenes101); each `video.hevc` is then decoded frame-by-frame, forward-only. `global_position` is expressed in a per-segment local frame (ECEF-axis-aligned, origin at the segment start), so absolute X/Y/Z and heading are segment-relative; the ego-relative past/future trajectories are unaffected. Native rate is 20 Hz — use `--frame_stride` to subsample and bound output volume.

³ TruckDrive is a long-range highway **heavy-truck** dataset (Torc Robotics / Princeton, CVPR 2026). Its 8 MP surround rig has **11 cameras** — more than the eight canonical `CameraDirection` slots — so each camera is mapped to the canonical surround member matching its facing (`FRONT`, `FRONT_LEFT`, `SIDE_LEFT`, `REAR_LEFT`, …) wherever one fits, with dedicated members added only for the genuinely-extra views the eight slots can't name (the forward telephoto pair `FRONT_{LEFT,RIGHT}_NARROW` and the rear-facing side pair `SIDE_{LEFT,RIGHT}_BACK`). `lidar_pc` is the seven-sensor **Aeva FMCW** joint cloud (xyz kept, in the ego frame); `detections_3d` are the tracked 3D boxes in the ego frame, with the ego vehicle's own cab/trailer and `DontCare` groups excluded per the paper taxonomy. The ego pose (PPK + LiDAR-SLAM) drives the past/future trajectory. Short-range Ouster lidar, 4D radar, accumulated GT depth and lane lines have no StandardE2E target yet and are not ingested. Frames are matched across sensors by their synchronization key. **The dataset ships as per-scene, per-modality zips and must be extracted first** — use [`scripts/extract_truckdrive.sh`](scripts/extract_truckdrive.sh), or [`scripts/prepare_dataset_truckdrive.sh`](scripts/prepare_dataset_truckdrive.sh) to extract and preprocess in one step.

⁴ View-of-Delft (VoD) (TU Delft, IEEE RA-L 2022) is a compact **urban** dataset whose distinctive sensor is a **3+1D radar**. StandardE2E ingests its single **front camera** (`CameraDirection.FRONT`, 1936×1216 pinhole), the 64-layer **Velodyne** `lidar_pc` (xyz in the ego/`velodyne` frame; per-point reflectance dropped) and KITTI-format `detections_3d` mapped from camera coordinates into the ego frame — each of VoD's 13 classes folded into the coarse `DetectionType` taxonomy (the two-wheeler family `bicycle`/`Cyclist`/`rider`/`moped_scooter`/`motor` → `BICYCLE`; `Car`/`truck`/`vehicle_other` → `VEHICLE`; static/ambiguous boxes → `UNKNOWN`; `DontCare` dropped). Box yaw is VoD's KITTI rotation about the LiDAR **−Z** axis (camera-x zero-reference), so the ego-frame heading is `−(rotation + π/2)`. One **frame = one keyframe**; one **segment = one recording scene** (`delft_*`), grouped via the official scene table so the per-segment past/future ego trajectory (from the per-frame `mapToCamera` pose) never spans two recordings. The **3+1D radar** (`radar` / `radar_3frames` / `radar_5frames`) has no StandardE2E modality yet and is not ingested; per-frame **timestamps are synthesised** at the 10 Hz LiDAR-lead rate (the detection release ships none); the **test** split has sensor data but no labels (so no detections). **The dataset ships as zips and must be extracted first** — use [`scripts/extract_vod.sh`](scripts/extract_vod.sh) (lidar tree only, with an optional track-id overlay), or [`scripts/prepare_dataset_vod.sh`](scripts/prepare_dataset_vod.sh) to extract and preprocess in one step.

⁵ nuScenes (Motional, CVPR 2020) is the de-facto surround-view E2E / BEV benchmark: 1000 ~20 s scenes, a **6-camera** surround rig (1600×900), a 32-beam `LIDAR_TOP` and densely annotated 3D boxes at 2 Hz keyframes (one **frame = one keyframe sample**, one **segment = one scene**). The six `CAM_*` channels map onto the canonical `CameraDirection` members; `lidar_pc` is the `LIDAR_TOP` cloud (xyz, in the ego frame); `detections_3d` are the `sample_annotation` boxes transformed from the global frame into the ego frame, each `category_name` folded into the coarse `DetectionType` taxonomy (nuScenes doesn't annotate the ego vehicle, so nothing is excluded). The ego pose drives the past/future trajectory. **HD map**: the vector **map-expansion** (lane centers recovered from the arcline paths, lane/road dividers, crossings, walkways, stop lines, drivable area, intersections) is translated to the unified `MapElementType` taxonomy in the ego frame and rasterised to BEV by `HDMapBEVAdapter`, **when** the separate `nuScenes-map-expansion-v1.3` pack is unzipped into `<dataroot>/maps/` (absent from the base download; the HD map is skipped if it isn't there). `--split` is an official nuScenes label that also selects the metadata version (`mini_train`/`mini_val` → `v1.0-mini`, `train`/`val` → `v1.0-trainval`, `test` → `v1.0-test`); the **test** split ships no annotations. nuScenes is read **directly from the JSON tables** — the `nuscenes-devkit` is *not* a runtime dependency (it pins `numpy<2`, conflicting with this project's numpy 2.x), so the split scene-lists and the lane-arcline discretization are vendored from it (Apache-2.0). The 5 radars have no StandardE2E target yet. A **partially-downloaded** trainval converts cleanly — scenes whose sensor blob isn't on disk yet are skipped. The release ships as `.tgz` archives and **must be extracted first** — use [`scripts/extract_nuscenes.sh`](scripts/extract_nuscenes.sh), or [`scripts/prepare_dataset_nuscenes.sh`](scripts/prepare_dataset_nuscenes.sh) to extract and preprocess in one step.

⁶ KITScenes Multimodal — the StandardE2E dataset key is `kitscenes_multimodal` (distinct from its long-tail sibling **KITScenes LongTail**, dataset key `kitscenes_longtail`, footnote ⁷) — (KIT / MRT, arXiv:2606.02956) is a large **European urban** dataset (~1000 scenes at 10 Hz) whose headline annotation is a dense, georeferenced **Lanelet2 HD map**. StandardE2E ingests its six surround **ring** cameras (`camera_ring_*` → the canonical `CameraDirection` surround members; pinhole `K`, `T_ego_from_camera` extrinsics from the calib's `T_to_reference`), the 128-beam **`lidar_top`** `lidar_pc` (xyz de-discretized from its int32 storage with the invalid-return sentinels dropped, in the ego/`base_frame` whose origin is `lidar_top`), and the **HD map**: each scene's Lanelet2 OSM (`maps/map.osm`) is parsed directly and translated to the unified `MapElementType` taxonomy — lanelet road/emergency/bicycle → `LANE_CENTER` (centerline from the left/right bound midpoints), `crosswalk` → `CROSSWALK`, `line_thin`/`line_thick`/`bike_marking` → `LANE_BOUNDARY`, `curbstone`/`road_border` → `ROAD_EDGE`, `stop_line` → `STOP_LINE`, `traffic_light*` → `TRAFFIC_LIGHT` — then cropped to an ego-centric ROI per frame and rasterised to BEV by `HDMapBEVAdapter`. The georeferencing needs no GNSS reconciliation: `poses.txt` is already in the Lanelet2 map-local frame (UTM zone 32N minus the `origin.json` anchor), verified by the ego trajectory lying on the map's node cloud; the map is projected with `pyproj` (no `lanelet2` runtime dependency — it pins `numpy<2`, like the nuScenes devkit). One **frame = one synchronized 10 Hz snapshot**; one **segment = one scene** (a UUID directory); the reference timeline gives per-frame timestamps and `poses.txt` the per-frame ego pose, which drives the past/future trajectory. KITScenes ships **no 3D object boxes** (the HD map is its annotation product), so `detections_3d` is always empty. Not ingested: the three "base" cameras (the hi-res front-center + the rectified stereo pair), the other six LiDARs, the three imaging radars and the GNSS/INS streams. Splits follow the official geo-disjoint folders (`train` / `val` / `test` / `test_e2e` / `overlap_train_val`); a flat layout (e.g. the single-scene sample) processes every scene with `--split` as the output label. **The release ships as per-split tarballs on Hugging Face (`KIT-MRT/KITScenes-Multimodal`) and must be downloaded and extracted first** — use [`scripts/extract_kitscenes.sh`](scripts/extract_kitscenes.sh), or [`scripts/prepare_dataset_kitscenes_multimodal.sh`](scripts/prepare_dataset_kitscenes_multimodal.sh) to extract and preprocess in one step.

⁷ KITScenes LongTail (dataset key `kitscenes_longtail`; KIT / MRT, arXiv:2603.23607) is the **long-tail / reasoning-traces** sibling of KITScenes Multimodal — ~1000 9 s scenarios deliberately filtered for **rare events** (adverse weather, construction, night, road closures, overtaking / lane changes). Unlike Multimodal's per-scene sensor directories it ships as Hugging Face `datasets` **parquet**, one **scenario per row**, and carries a different modality set. Each scenario is a 360° multi-view video, so it is **unrolled into one frame per timestep** (5 Hz; one **frame = one timestep**, one **segment = one scenario**, ~21 frames over the 4 s observation window): per frame the six surround **ring** cameras at that timestep (mapped to the canonical `CameraDirection` surround members with the fixed-rig pinhole `K` and `T_ego_from_camera` from the README's `(K, R, t)`, where `R` is ego→camera so the extrinsics are `inv(se3(R, t))`), the high-level **`driving_instruction`** folded into the coarse `Intent` (`left`→`GO_LEFT`, `right`→`GO_RIGHT`, `straight`→`GO_STRAIGHT`, else `UNKNOWN`; the raw instruction kept in `aux_data`), and **`past_states`** — the ego history up to that timestep, re-expressed **ego-relative to the frame** (ego at the origin, facing +x along its path tangent), metres (x forward, y left). The **prediction targets** — the 5 s **expert** future (`future_states`, 25 pts) and the **counterfactual** futures (`wrong_speed` / `neglect_instruction` / `off_road` / `crash`) as **`preference_trajectory`** (only the ones present) — are attached to the **`t=0` (last) frame** only (marked `is_prediction_frame` in the index). It ships **no lidar, HD map or 3D boxes**. The multilingual **reasoning traces** (English / Spanish / Chinese — situational awareness + per-phase acceleration/steering rationale) are surfaced (English) into `aux_data`. Note: the released **`test`** split withholds the future trajectories (eval ground truth — encoded as a `[[-100, -100]]` sentinel); only **`train`** / **`train_raw`** carry real expert + counterfactual futures. The **`_raw`** splits are the native-resolution frames; the non-`raw` splits are the processed frames the vendored intrinsics match. **The release ships as gated Hugging Face parquet (`KIT-MRT/KITScenes-LongTail`) and must be downloaded first** — see [`scripts/prepare_dataset_kitscenes_longtail.sh`](scripts/prepare_dataset_kitscenes_longtail.sh).

⁸ NATIX Multi-Camera (dataset key `natix_multicam`; its curated edge-case sibling is supported as `natix_edgecase` — footnote ⁹, mirroring the KITScenes variant naming — and NATIX has also announced a telemetry-included release) is a **crowd-sourced** real-world driving dataset: 100 h of **Tesla-dashcam** surround footage from everyday, non-expert drivers in Switzerland and the United States — ~1-minute mp4 clips at ~36 fps plus per-clip **consumer-grade GPS** metadata (1-10 Hz), per-trip camera calibration and trip-level metadata. One **frame = one front-camera GPS fix** (the ~36 fps video between fixes carries no pose and is not emitted; `--frame_stride` subsamples further); one **segment = one trip piece** (`<trip-id>_<n>[_seqNN]` — continuous minutes), its clips concatenated chronologically with the wall-clock CSV timestamps localized via `trip_insight.json`'s IANA timezone (verified against `startEpochMs`). **Cameras** (4-camera rigs: `FRONT`/`REAR`/`LEFT`/`RIGHT`; 6-camera rigs add the pillar pair and name the repeaters explicitly) map to the canonical `CameraDirection` by **facing**, verified from the calibration rotations: in 4-camera trips `LEFT`/`RIGHT` *are* the backward-facing repeaters → `REAR_LEFT`/`REAR_RIGHT` (not `SIDE_*`), and the pillar cameras face forward-side → `FRONT_LEFT`/`FRONT_RIGHT`. Pinhole `K` + Brown-Conrady distortion are taken as shipped (the front camera can be natively 2896×1876 while the trip's standard cameras are 1448×938 — its `K` already matches the native size); extrinsics are `T_ego_from_camera` in the **optical** frame, converted from the `ground_nominal` (FLU ego) body-frame rotations, centimetres → metres. Because per-camera streams differ in frame rate and duration, each camera is matched to the fix **by timestamp** (nearest CSV row → its `frame_number`); a camera with no usable match (missing files, >1 s offset) is absent from that frame's `cameras`. The **ego pose** is GPS-derived: lat/lon through a per-segment azimuthal-equidistant local east/north frame (anchored at the segment's first fix; z = 0 — no altitude), yaw from `heading_deg` (clockwise-from-north → FLU), with unreliable headings (missing, or ≤0.5 m/s where GPS noise dominates) held from the last reliable fix; `global_position` also carries the fix **speed**, and the pose drives the past/future trajectories. `extra_index_data` carries trip / country / region / camera_count for filtering. It ships **no lidar, HD map, 3D boxes or driving command**, and **no canonical split** (`--split` is a passthrough label). Mind the source-data quality bounds (documented, not corrected): GPS fix accuracy is tens of metres, video/GPS sync error 0-1 s (extremes to 3 s), and calibrations are **nominal per vehicle model**, not per-vehicle calibrated. **The release needs no extraction** — the gated Hugging Face repo (`natix-network-org/natix-multi-camera-driving-dataset`) carries a ~20-minute `dataset-sample/`; the full ~1.28 TB is pulled from a Cloudflare R2 bucket with credentials granted on access approval — see [`scripts/prepare_dataset_natix_multicam.sh`](scripts/prepare_dataset_natix_multicam.sh).

⁹ NATIX Edge Case (dataset key `natix_edgecase`) is the **curated edge-case sibling** of NATIX Multi-Camera: rare, challenging real-world scenarios — construction zones, adverse weather, road-surface deterioration, illegal maneuvers, obstructions — crowd-sourced from the same decentralized Tesla-dashcam network. The first public release is **20 minutes / 86 mp4 clips** across six US states (4- and 6-camera rigs) with **21 VLM-annotated events**. Footage, GPS metadata, calibration and folder layout are **identical to `natix_multicam`** (same frame/segment definitions, camera mapping, GPS ego pose and data-quality bounds — everything in footnote ⁸ applies; the processor is a subclass). What the release adds are the **edge-case annotations** (`data/edge-case.json`): per annotated clip, one or more events, each with a human-readable `label` (may be empty), a `[start_sec, end_sec]` window in **video seconds from clip start**, and a structured **VLM `ai_analysis`** (event classification, visual evidence, context, agentic validation, final detected event). Every emitted frame carries the events covering its timestamp **inside its `.npz`**, as `aux_data["edge_case_events"]` — a list (empty outside events) of one plain dict per covering event with the verbatim `label`, `start_sec` / `end_sec` window and the complete `ai_analysis` object — and, flattened into the **index**, `extra_edge_case_count` / `extra_edge_case` (the summaries; a missing `label` falls back to the VLM `EVENT CLASSIFICATION`) — so in-event frames are filterable via `extra_edge_case_count > 0` without touching the npz, and the full analysis text travels with the frame itself (see the [quickstart](https://standarde2e.readthedocs.io/en/latest/quickstart.html) for an access snippet). Windows map onto the frame timeline through each clip's front CSV (the epoch of video frame 1, extrapolated when the CSV's leading rows are missing — observed in the release). The annotations are **VLM-generated best-effort** (integer-second windows, model-written descriptions) — treat them as guidance, not ground truth. One further observed GPS degradation (passed through as-shipped): a trip's fixes can **freeze** (lat/lon stuck, speed 0) for tens of seconds while the footage keeps moving — observed on one release trip, overlapping its event window — leaving the pose-derived past/future trajectories degenerate over that stretch even though the cameras are fine. **The release needs no extraction** — the gated Hugging Face repo (`natix-network-org/natix-edge-case-driving-dataset`, ~3.3 GB) carries everything under `data/` — see [`scripts/prepare_dataset_natix_edgecase.sh`](scripts/prepare_dataset_natix_edgecase.sh).

## 🚀 Key Features

- **Unified Dataset Interface**: Work with multiple datasets through a single API
- **Multimodal Support**: Cameras, LiDAR (point cloud + BEV histogram), HD maps (BEV raster), trajectories, detections and more
- **Flexible Preprocessing**: Configurable pipelines with standardization and augmentation
- **Lazy modality loading**: Configured adapters declare what they consume; no decoding work is done for modalities no adapter reads
- **Trajectory Management**: Advanced handling of time-series vehicle data
- **PyTorch Integration**: Ready-to-use datasets and dataloaders

## 📝 Quick Start & Examples
### Notebooks

- [intro_tutorial.ipynb](notebooks/intro_tutorial.ipynb) - Introduction to StandardE2E framework
- [containers.ipynb](notebooks/containers.ipynb) - Working with data containers
- [multi_dataset_training_and_filtering.ipynb](notebooks/multi_dataset_training_and_filtering.ipynb) - Multi-dataset training and filtering
- [creating_custom_adapter.ipynb](notebooks/creating_custom_adapter.ipynb) - Creating custom dataset adapters

### Code Examples

Run from the project root so `uv run` uses the project environment. If you use pip/conda instead, activate your env and use `python` in place of `uv run python`.

1. **Preprocess Waymo End-to-end dataset** - Convert raw dataset to standardized format ([`dataset_preprocessing.py`](examples/dataset_preprocessing.py))
    ```bash
    uv run python examples/dataset_preprocessing.py \
      --e2e_dataset_path E2E_DATASET_PATH \
      --split {training,val,test} \
      --processed_data_path PROCESSED_DATA_PATH
    ```
2. **Train your model** - End-to-end training with multimodal data ([`very_simple_training.py`](examples/very_simple_training.py)). This example illustrates iteration over the preprocessed dataset. Also, in this example for validation we use 2 DataLoaders - full validation split and filtered validation split that only contains samples with preferred trajectories.
    ```bash
    uv run python examples/very_simple_training.py --processed_data_path PROCESSED_DATA_PATH
    ```
3. **Create a unified DataLoader**: This example shows how to process 2 different datasets within same DataLoader. First, please do preprocessing for `Waymo E2E` and `Waymo Perception` datasets in order to utilize them in the DataLoader with the script ([`prepare_datasets_waymo_e2e_perception.sh`](scripts/prepare_datasets_waymo_e2e_perception.sh)).

    The script [`creating_unified_dataloader.py`](examples/creating_unified_dataloader.py) created a unified dataloader that iterates over both `Waymo E2E` and `Waymo Perception` in one epoch providing consistent data structure.

    ```bash
    uv run python examples/creating_unified_dataloader.py --processed_data_path PROCESSED_DATA_PATH
    ```
4. **Visualize processed output** - Render any processed `<dataset>/<split>` folder to per-scene MP4s, auto-detecting whichever modalities the frames carry: a camera mosaic (surround grid or stitched panorama) plus a co-registered BEV panel (HD-map / lidar / detection rasters, lidar points, 3D boxes, past/future/preference trajectories, ego). Playback speed defaults to the rate inferred from the frame timestamps; `--fps` overrides. Select scenes with `--num-scenes N` or repeatable `--scene-id ID` (default: first scene).
    ```bash
    uv run python -m standard_e2e.visualization.visualize_processed \
      PROCESSED_DATA_PATH/DATASET_NAME/SPLIT --num-scenes 2 --out visualizations
    ```
5. **Add a new dataset adapter** - Guide for adding support for new datasets ([`adding_new_dataset.md`](standard_e2e/caching/src_datasets/adding_new_dataset.md))


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this project useful, you can support it by giving it a ⭐, or by contributing with your PRs / issues / feature requests. Also, if you use this project, you can greatly support it by citing our paper ([arXiv:2606.04271](https://arxiv.org/abs/2606.04271)):
```bibtex
@misc{konev2026standarde2e,
  title={StandardE2E: A Unified Framework for End-to-End Autonomous Driving Datasets},
  author={Stepan Konev},
  year={2026},
  eprint={2606.04271},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.04271}
}
```
and using the badge [![StandardE2E](https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/StandardE2E_gh_badge_dark.svg)](https://github.com/stepankonev/StandardE2E)

Markdown

```markdown
[![StandardE2E](https://raw.githubusercontent.com/stepankonev/StandardE2E/refs/heads/main/assets/StandardE2E_gh_badge_dark.svg)](https://github.com/stepankonev/StandardE2E)
```

HTML

```html
<a href="https://github.com/stepankonev/StandardE2E">
  <img src="https://raw.githubusercontent.com/stepankonev/StandardE2E/refs/heads/main/assets/StandardE2E_gh_badge_dark.svg" alt="StandardE2E"/>
</a>
```
