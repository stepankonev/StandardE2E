<div align="center">
  <img src="assets/standard_e2e_logo_contrast.png" alt="StandardE2E Logo" width="400"/>
  
  <p><em>A framework for unified end-to-end autonomous driving datasets processing</em></p>

  ![Python versions](https://img.shields.io/badge/Python-3.12%20%7C%203.13-informational?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
  [![Tests](https://github.com/stepankonev/StandardE2E/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/stepankonev/StandardE2E/actions/workflows/tests.yml)
  [![codecov](https://codecov.io/github/stepankonev/StandardE2E/graph/badge.svg?token=3MWJNB10OO)](https://codecov.io/github/stepankonev/StandardE2E)
  [![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
  [![mypy](https://img.shields.io/badge/mypy-checked-2A6DB2?logo=mypy&logoColor=white)](http://mypy-lang.org/)
  [![StandardE2E](assets/StandardE2E_gh_badge_dark.svg)](https://github.com/stepankonev/StandardE2E)
  
</div>

<div align="center">

  [![Docs](https://readthedocs.org/projects/standard-e2e/badge/?version=latest)](https://standard-e2e.readthedocs.io/en/latest/)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/vJnQNcQGQ8)

</div>

StandardE2E provides a consistent interface for preprocessing, loading, and training with multimodal data from various end-to-end autonomous driving datasets. It standardizes the complex process of working with different dataset formats, allowing researchers to focus on model development rather than data engineering.

> ⚠️ **Early Beta**
>
> This project is in **early beta**. **Read the Docs** and the **PyPI** package are **not yet available**, but will be available soon.


![StandardE2E Architecture](assets/standard_e2e_scheme.png)

---

## 📖 Documentation

- Latest docs: https://standard-e2e.readthedocs.io/en/latest/

## 📦 Installation

### Option 1: From PyPI (Recommended for Users)
```bash
pip install standard-e2e
```

### Option 2: Manual Development Setup
```bash
# Create a new conda environment
conda create -n standard_e2e python=3.12
conda activate standard_e2e

# Install the package in development mode
pip install -e .
```

## Plan for E2E Autonomous Driving Datasets Support


| Dataset | Cameras | Lidar | HD Map | Detections | Driving Command | Preference Trajectories |
|---------|---------|-------|--------|------------|-----------------|-------------------------|
| [Waymo End-to-end](https://waymo.com/open/data/e2e/) ![](https://img.shields.io/badge/supported-darkgreen) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ❌ | ❌ | ✅ | ✅ |
| [Waymo Perception](https://waymo.com/open/data/perception/) ![](https://img.shields.io/badge/WIP-orange) | ![](https://img.shields.io/badge/semicircle-orange) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Navsim](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ✅ | ❌ |
| [WayveScenes101](https://wayve.ai/science/wayvescenes101) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/semicircle-orange) | ✅ | ❌ | ❌ | ❌ | ❌ |
| [Argoverse 2 Sensor](https://www.argoverse.org/av2.html#sensor-link) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/circle-darkgreen) | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Lidar](https://www.argoverse.org/av2.html#lidar-link) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Argoverse 2 Map Change](https://www.argoverse.org/av2.html#mapchange-link) ![](https://img.shields.io/badge/TBD-gray) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) ![](https://img.shields.io/badge/TBD-gray) | ![](https://img.shields.io/badge/front-darkred) + 2 x ![](https://img.shields.io/badge/side%20fisheye-blue) | ✅ | ❓ | ❓ | ❓ | ❓ |
| [Waymo Motion Prediction](https://waymo.com/open/data/motion/) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Argoverse 2 Motion Forecasting [?]](https://www.argoverse.org/av2.html#forecasting-link) ![](https://img.shields.io/badge/TBD-gray) | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |

## 🚀 Key Features

- **Unified Dataset Interface**: Work with multiple datasets through a single API
- **Multimodal Support**: Cameras, LiDAR ![](https://img.shields.io/badge/WiP-orange), HD maps ![](https://img.shields.io/badge/WiP-orange), trajectories, detections and more
- **Flexible Preprocessing**: Configurable pipelines with standardization and augmentation  
- **Trajectory Management**: Advanced handling of time-series vehicle data
- **PyTorch Integration**: Ready-to-use datasets and dataloaders


## 📝 Quick Start & Examples
### Notebooks

- [intro_tutorial.ipynb](notebooks/intro_tutorial.ipynb) - Introduction to StandardE2E framework
- [containers.ipynb](notebooks/containers.ipynb) - Working with data containers
- [multi_dataset_training_and_filtering.ipynb](notebooks/multi_dataset_training_and_filtering.ipynb) - Multi-dataset training and filtering
- [creating_custom_adapter.ipynb](notebooks/creating_custom_adapter.ipynb) - Creating custom dataset adapters

### Code Examples
1. **Preprocess Waymo End-to-end dataset** - Convert raw dataset to standardized format ([`dataset_preprocessing.py`](examples/dataset_preprocessing.py))
    ```
    python examples/dataset_preprocessing.py \
      --e2e_dataset_path E2E_DATASET_PATH \
      --split {training,val,test} \
      --processed_data_path PROCESSED_DATA_PATH
    ```
2. **Train your model** - End-to-end training with multimodal data ([`very_simple_training.py`](examples/very_simple_training.py)). This example illustrates iteration over the preprocessed dataset. Also, in this example for validation we use 2 DataLoaders - full validation split and filtered validation split that only contains samples with preferred trajectories.
    ```
    python examples/very_simple_training.py --processed_data_path PROCESSED_DATA_PATH
    ```
3. **Create a unified DataLoader**: This example shows how to process 2 different datasets within same DataLoader. First, please do preprocessing for `Waymo E2E` and `Waymo Perception` datasets in order to utilize them in the DataLoader with the script ([`prepare_datasets_waymo_e2e_perception.sh`](scripts/prepare_datasets_waymo_e2e_perception.sh)).

    The script [`creating_unified_dataloader.py`](examples/creating_unified_dataloader.py) created a unified dataloader that iterates over both `Waymo E2E` and `Waymo Perception` in one epoch providing consistent data structure.

    ```
    python examples/creating_unified_dataloader.py --processed_data_path PROCESSED_DATA_PATH
    ```
4. **Add a new dataset adapter** - Guide for adding support for new datasets ([`adding_new_dataset.md`](standard_e2e/caching/src_datasets/adding_new_dataset.md))


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this project useful, you can support it by giving it a ⭐, or by contributing with your PRs / issues / feature requests. Also, if you use this project, you can greatly support it by citing
```bibtex
@software{standarde2e,
  title={StandardE2E: A Unified Framework for Autonomous Driving Dataset Management},
  author={stepankonev},
  year={2025},
  url={https://github.com/stepankonev/StandardE2E}
}
```
and using the badge [![StandardE2E](assets/StandardE2E_gh_badge_dark.svg)](https://github.com/stepankonev/StandardE2E)

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
