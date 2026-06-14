<div align="center">
  <img src="https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/standard_e2e_logo_contrast.png" alt="StandardE2E Logo" width="400"/>
  
  <p><em>A framework for unified end-to-end autonomous driving datasets processing</em></p>

  ![Python versions](https://img.shields.io/badge/Python-3.12-informational?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
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
| [Argoverse 2 Map Change](https://www.argoverse.org/av2.html#mapchange-link) ![](https://img.shields.io/badge/TBD-gray) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) + 2 x ![](https://img.shields.io/badge/side%20fisheye-blue) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [Waymo Motion Prediction](https://waymo.com/open/data/motion/) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Motion Forecasting [?]](https://www.argoverse.org/av2.html#forecasting-link) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |

¹ WayveScenes101 ships no sensor lidar; the `lidar_pc` modality is populated from the per-scene **COLMAP SfM** point cloud, transformed into each frame's ego frame and range-clipped so it flows through the lidar adapters. It is photogrammetric (sparse, up-to-scale), not a sensor measurement. Cameras are 5 fisheye views (forward + side arc); the ego past/future trajectory is derived from the COLMAP poses. Scenes must be extracted from the distributed `scene_<NNN>.zip` archives before processing.

² comma2k19 is a single forward-facing 20 Hz camera (comma EON, 1164×874, pinhole) with a fused GNSS/IMU ego pose; the ego past/future trajectory is derived from the global ECEF poses and `global_position` additionally carries the ego **speed**. It ships no lidar, HD map, 3D boxes, or driving command. Segments must be extracted from the distributed `Chunk_*.zip` archives before processing (as with WayveScenes101); each `video.hevc` is then decoded frame-by-frame, forward-only. `global_position` is expressed in a per-segment local frame (ECEF-axis-aligned, origin at the segment start), so absolute X/Y/Z and heading are segment-relative; the ego-relative past/future trajectories are unaffected. Native rate is 20 Hz — use `--frame_stride` to subsample and bound output volume.

³ TruckDrive is a long-range highway **heavy-truck** dataset (Torc Robotics / Princeton, CVPR 2026). Its 8 MP surround rig has **11 cameras** — more than the eight canonical `CameraDirection` slots — so each camera is mapped to the canonical surround member matching its facing (`FRONT`, `FRONT_LEFT`, `SIDE_LEFT`, `REAR_LEFT`, …) wherever one fits, with dedicated members added only for the genuinely-extra views the eight slots can't name (the forward telephoto pair `FRONT_{LEFT,RIGHT}_NARROW` and the rear-facing side pair `SIDE_{LEFT,RIGHT}_BACK`). `lidar_pc` is the seven-sensor **Aeva FMCW** joint cloud (xyz kept, in the ego frame); `detections_3d` are the tracked 3D boxes in the ego frame, with the ego vehicle's own cab/trailer and `DontCare` groups excluded per the paper taxonomy. The ego pose (PPK + LiDAR-SLAM) drives the past/future trajectory. Short-range Ouster lidar, 4D radar, accumulated GT depth and lane lines have no StandardE2E target yet and are not ingested. Frames are matched across sensors by their synchronization key. **The dataset ships as per-scene, per-modality zips and must be extracted first** — use [`scripts/extract_truckdrive.sh`](scripts/extract_truckdrive.sh), or [`scripts/prepare_dataset_truckdrive.sh`](scripts/prepare_dataset_truckdrive.sh) to extract and preprocess in one step.

⁴ View-of-Delft (VoD) (TU Delft, IEEE RA-L 2022) is a compact **urban** dataset whose distinctive sensor is a **3+1D radar**. StandardE2E ingests its single **front camera** (`CameraDirection.FRONT`, 1936×1216 pinhole), the 64-layer **Velodyne** `lidar_pc` (xyz in the ego/`velodyne` frame; per-point reflectance dropped) and KITTI-format `detections_3d` mapped from camera coordinates into the ego frame — each of VoD's 13 classes folded into the coarse `DetectionType` taxonomy (the two-wheeler family `bicycle`/`Cyclist`/`rider`/`moped_scooter`/`motor` → `BICYCLE`; `Car`/`truck`/`vehicle_other` → `VEHICLE`; static/ambiguous boxes → `UNKNOWN`; `DontCare` dropped). Box yaw is VoD's KITTI rotation about the LiDAR **−Z** axis (camera-x zero-reference), so the ego-frame heading is `−(rotation + π/2)`. One **frame = one keyframe**; one **segment = one recording scene** (`delft_*`), grouped via the official scene table so the per-segment past/future ego trajectory (from the per-frame `mapToCamera` pose) never spans two recordings. The **3+1D radar** (`radar` / `radar_3frames` / `radar_5frames`) has no StandardE2E modality yet and is not ingested; per-frame **timestamps are synthesised** at the 10 Hz LiDAR-lead rate (the detection release ships none); the **test** split has sensor data but no labels (so no detections). **The dataset ships as zips and must be extracted first** — use [`scripts/extract_vod.sh`](scripts/extract_vod.sh) (lidar tree only, with an optional track-id overlay), or [`scripts/prepare_dataset_vod.sh`](scripts/prepare_dataset_vod.sh) to extract and preprocess in one step.

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
4. **Add a new dataset adapter** - Guide for adding support for new datasets ([`adding_new_dataset.md`](standard_e2e/caching/src_datasets/adding_new_dataset.md))


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
