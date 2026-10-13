<div align="center">
  <img src="https://raw.githubusercontent.com/stepankonev/StandardE2E/main/assets/standard_e2e_logo_contrast.png" alt="StandardE2E Logo" width="400"/>
  
  <p><em>A framework for unified end-to-end autonomous driving datasets processing</em></p>

  ![Python versions](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-informational?logo=python&logoColor=white)
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

The repository root is the **training environment**: StandardE2E with PyTorch, and no
dataset SDKs. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/stepankonev/StandardE2E.git
cd StandardE2E
uv sync
```

In your own project, `pip install standard-e2e` (or `uv add standard-e2e`) installs the
same thing. For development, add the test and lint tools and run the tests (each
dataset's tests skip where its SDK is not installed; they run in that dataset's
environment, see [`datasets/`](datasets/README.md)):

```bash
uv sync --all-extras
uv run pytest tests/
```

## 🗂️ Preprocessing a dataset

Every dataset has a folder under [`datasets/`](datasets/README.md) with its description,
its own isolated environment (only the SDK it needs: TensorFlow for Waymo, `av2` for
Argoverse 2, nuplan-devkit for NAVSIM, ...) and a preprocessing pipeline that
[stepix](https://github.com/stepankonev/stepix) runs. Set the paths in the dataset's
README, then:

```bash
uvx stepix run datasets/waymo_e2e/README.md
```

Where the data can be fetched by a script (Waymo, Argoverse 2, nuScenes mini, comma2k19,
KITScenes, NATIX Edge Case), setting `download = true` in the README makes the pipeline
download the split first, with the dataset's own tool and your access.

## ⬆️ Upgrading from 0.0.x

0.1.0 splits StandardE2E into the training environment (the repository root) and one
environment per dataset with that dataset's SDK. Data processed with 0.0.x loads
unchanged, except for the camera extrinsics of four datasets (last point); code needs
these changes:

- **PyTorch-side names moved to `standard_e2e.training`**, the only sub-package that
  imports PyTorch (`import standard_e2e` no longer does):

  | 0.0.x | 0.1.0 |
  |---|---|
  | `standard_e2e.UnifiedE2EDataset`, `standard_e2e.unified_dataset.UnifiedE2EDataset` | `standard_e2e.training.UnifiedE2EDataset` |
  | `standard_e2e.dataset_utils.augmentation.MultipleFramesImageAugmentation` | `standard_e2e.training.MultipleFramesImageAugmentation` |
  | `TransformedFrameDataBatch`, `BatchedTrajectory`, `BatchedLidarPointCloud`, `BatchedFrameDetections3D`, `collate_modalities` from `standard_e2e.data_structures` | the same names from `standard_e2e.training` |

- **TensorFlow-side names moved to modules of their own**, importable where TensorFlow is
  installed (the Waymo environments):

  | 0.0.x | 0.1.0 |
  |---|---|
  | `standard_e2e.caching.TFRecSourceDatasetConverter` | `standard_e2e.caching.tfrec_source_dataset_converter.TFRecSourceDatasetConverter` |
  | `standard_e2e.utils.image_utils.waymo_fetch_images_from_frame`, `WAYMO_CAMERAS_ORDER` | the same names from `standard_e2e.utils.waymo_image_utils` |

- **Preprocessing runs in the dataset's environment.** TensorFlow and the other dataset
  SDKs are no longer installed with StandardE2E. Run the dataset's pipeline
  (`uvx stepix run datasets/<name>/README.md`), or the same CLI or your own script in that
  environment: `uv run --project datasets/<name> python -m standard_e2e.caching.process_source_dataset <name> ...`.
  Outside a checkout, install the SDK listed in `datasets/<name>/pyproject.toml`.

- **Camera extrinsics are in the OpenCV optical frame for every dataset.**
  `CameraData.extrinsics` is `T_ego_from_camera` with camera axes x right, y down,
  z forward (the frame `intrinsics` projects from). In 0.0.x, Waymo Open E2E, Waymo Open
  Perception, WayveScenes101 and comma2k19 stored an x-forward / y-left / z-up camera
  frame instead, so `K` could not project through their `T`. Re-preprocess data of these
  four datasets that was made with the `cameras_identity_adapter`, or convert it in place
  with `T_new = T_old @ FLU_FROM_OPTICAL` (`standard_e2e.utils.FLU_FROM_OPTICAL`). Data
  made with the `pano_adapter` carries no extrinsics and is unaffected.

## Supported datasets


| Dataset | Cameras | Lidar | HD Map | Detections | Driving Command | Preference Trajectories |
|---------|---------|-------|--------|------------|-----------------|-------------------------|
| [Waymo End-to-end](datasets/waymo_e2e/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ❌ | ❌ | ❌ | ✅ | ✅ |
| [Waymo Perception](datasets/waymo_perception/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/semicircle-orange) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Navsim](datasets/navsim/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ✅ | ❌ |
| [WayveScenes101](datasets/wayve_scenes/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/semicircle-orange) | ![](https://img.shields.io/badge/SfM-orange) | ❌ | ❌ | ❌ | ❌ |
| [Argoverse 2 Sensor](datasets/av2_sensor/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Lidar](datasets/av2_lidar/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [comma2k19](datasets/comma2k19/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/front-darkred) | ❌ | ❌ | ❌ | ❌ | ❌ |
| [TruckDrive](datasets/truckdrive/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ❌ | ✅ | ❌ | ❌ |
| [View-of-Delft](datasets/vod/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/front-darkred) | ✅ | ❌ | ✅ | ❌ | ❌ |
| [nuScenes](datasets/nuscenes/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [KITScenes Multimodal](datasets/kitscenes_multimodal/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITScenes LongTail](datasets/kitscenes_longtail/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ❌ | ❌ | ❌ | ✅ | ✅ |
| [NATIX Multi-Camera](datasets/natix_multicam/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/ring_4--6_cam-orange) | ❌ | ❌ | ❌ | ❌ | ❌ |
| [NATIX Edge Case](datasets/natix_edgecase/README.md) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/ring_4--6_cam-orange) | ❌ | ❌ | ❌ | ❌ | ❌ |
| [Argoverse 2 Map Change](https://www.argoverse.org/av2.html#mapchange-link) ![](https://img.shields.io/badge/TBD-gray) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) + 2 x ![](https://img.shields.io/badge/side%20fisheye-blue) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [Waymo Motion Prediction](https://waymo.com/open/data/motion/) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Motion Forecasting [?]](https://www.argoverse.org/av2.html#forecasting-link) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |

Each dataset's folder describes what StandardE2E extracts from it, its caveats, and how to run its pipeline; rows marked TBD are not supported yet.

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

The first three read data that a dataset's pipeline has already processed, so they run in
the training environment: `uv run --with jupyter jupyter lab`. The adapter notebook
preprocesses Waymo End-to-end data itself, so start it from that dataset's environment:
`uv run --project datasets/waymo_e2e --with jupyter jupyter lab`.

### Code Examples

Run from the project root. Training-side examples use the default environment (`uv run`); preprocessing runs in the dataset's own environment (`uv run --project datasets/<name>`).

1. **Preprocess Waymo End-to-end dataset** - Convert raw dataset to standardized format from Python, with custom adapters ([`dataset_preprocessing.py`](examples/dataset_preprocessing.py)); the [`datasets/waymo_e2e`](datasets/waymo_e2e/README.md) pipeline does the same from its config
    ```bash
    uv run --project datasets/waymo_e2e python examples/dataset_preprocessing.py \
      --e2e_dataset_path E2E_DATASET_PATH \
      --split {training,val,test} \
      --processed_data_path PROCESSED_DATA_PATH
    ```
2. **Train your model** - End-to-end training with multimodal data ([`very_simple_training.py`](examples/very_simple_training.py)). This example illustrates iteration over the preprocessed dataset. Also, in this example for validation we use 2 DataLoaders - full validation split and filtered validation split that only contains samples with preferred trajectories.
    ```bash
    uv run python examples/very_simple_training.py --processed_data_path PROCESSED_DATA_PATH
    ```
3. **Create a unified DataLoader**: This example shows how to process 2 different datasets within same DataLoader. First, preprocess `Waymo E2E` and `Waymo Perception` into the same output root with their pipelines ([`datasets/waymo_e2e`](datasets/waymo_e2e/README.md), [`datasets/waymo_perception`](datasets/waymo_perception/README.md)).

    The script [`creating_unified_dataloader.py`](examples/creating_unified_dataloader.py) created a unified dataloader that iterates over both `Waymo E2E` and `Waymo Perception` in one epoch providing consistent data structure.

    ```bash
    uv run python examples/creating_unified_dataloader.py --processed_data_path PROCESSED_DATA_PATH
    ```
4. **Visualize processed output** - Render any processed `<dataset>/<split>` folder to per-scene MP4s, auto-detecting whichever modalities the frames carry: a camera mosaic (surround grid or stitched panorama) plus a co-registered BEV panel (HD-map / lidar / detection rasters, lidar points, 3D boxes, past/future/preference trajectories, ego). Playback speed defaults to the rate inferred from the frame timestamps; `--fps` overrides. Select scenes with `--num-scenes N` or repeatable `--scene-id ID` (default: first scene).
    ```bash
    uv run python -m standard_e2e.visualization.visualize_processed \
      PROCESSED_DATA_PATH/DATASET_NAME/SPLIT --num-scenes 2 --out visualizations
    ```
5. **Add a new dataset adapter** - Guide for adding support for new datasets ([`adding_new_dataset.md`](standard_e2e/caching/src_datasets/adding_new_dataset.md)), plus its folder under [`datasets/`](datasets/README.md#adding-a-dataset)


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
