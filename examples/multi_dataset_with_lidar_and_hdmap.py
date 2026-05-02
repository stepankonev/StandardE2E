"""Showcase: combining Waymo E2E and Waymo Perception in one DataLoader.

Each batch mixes samples from both datasets. Modalities present in only
one of them (``LIDAR_PC`` and ``HD_MAP_BEV`` for Perception; ``INTENT``
and ``PREFERENCE_TRAJECTORY`` for E2E) are filled in by the matching
``ModalityDefaults`` handler so collate produces a uniform-shape tensor
for every modality regardless of which dataset a sample came from.

The YAML file (``configs/base.yaml`` by default) is the single source of
truth for both the preprocessing-time adapter list (and their params --
including the HD-map BEV extent / resolution) and the runtime feature /
label loaders. The script reads that file, rebuilds the
``HDMapBEVAdapter`` from its YAML params, and passes that instance to
``HDMapBEVDefaults.from_adapter(...)`` so the empty-tensor placeholders
for E2E samples have exactly the same shape as the rasters Perception
samples carry on disk.

Prerequisites:
    - Both Waymo E2E and Waymo Perception preprocessed under a common
      output root using this YAML's adapter list.
      ``scripts/prepare_datasets_waymo_e2e_perception.sh`` is the easy
      way to produce both.

Run from the repo root::

    uv run python examples/multi_dataset_with_lidar_and_hdmap.py \\
        --processed_data_path /mnt/nvme1/data/standard_e2e_test_waymo

Optional flags: ``--config_file``, ``--batch_size``, ``--num_batches``.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from standard_e2e import UnifiedE2EDataset
from standard_e2e.caching.adapters import HDMapBEVAdapter, get_adapters_from_config
from standard_e2e.dataset_utils.augmentation import MultipleFramesImageAugmentation
from standard_e2e.dataset_utils.modality_defaults import (
    HDMapBEVDefaults,
    IntentDefaults,
    LidarPointCloudDefaults,
    PreferredTrajectoryDefaults,
)
from standard_e2e.enums import Modality, TrajectoryComponent
from standard_e2e.indexing import get_multi_dataset_index
from standard_e2e.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_path",
        type=Path,
        required=True,
        help="Root containing waymo_e2e/ and waymo_perception/ preprocessed splits.",
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=Path("configs/multi_dataset_with_lidar_and_hdmap.yaml"),
        help="YAML config: preprocessing.adapters drives the BEV-adapter shape "
        "used by HDMapBEVDefaults; dataset.features/labels drives the "
        "FrameLoader specs.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(str(args.config_file))

    # Re-instantiate the YAML's preprocessing adapters here so the BEV
    # defaults' empty-tensor shape stays in sync with whatever the YAML
    # used at preprocessing time.
    preprocessing_adapters = get_adapters_from_config(
        config["preprocessing"]["adapters"]
    )
    bev_adapter = next(
        a for a in preprocessing_adapters if isinstance(a, HDMapBEVAdapter)
    )

    e2e_idx = pd.read_parquet(
        args.processed_data_path / "waymo_e2e" / "training" / "index.parquet"
    )
    perception_idx = pd.read_parquet(
        args.processed_data_path / "waymo_perception" / "training" / "index.parquet"
    )
    print(f"e2e: {len(e2e_idx)} frames, perception: {len(perception_idx)} frames")

    # Cap whichever index is much larger so batches actually mix the two.
    if len(e2e_idx) > len(perception_idx) * 4:
        e2e_idx = e2e_idx.sample(n=len(perception_idx), random_state=0).reset_index(
            drop=True
        )
    elif len(perception_idx) > len(e2e_idx) * 4:
        perception_idx = perception_idx.sample(
            n=len(e2e_idx), random_state=0
        ).reset_index(drop=True)

    unified_idx = get_multi_dataset_index([e2e_idx, perception_idx])

    dataset = UnifiedE2EDataset(
        index_data=unified_idx,
        processed_data_path=str(args.processed_data_path),
        regime="val",
        feature_loaders_config=config["dataset"]["features"],
        label_loaders_config=config["dataset"]["labels"],
        augmentations=[MultipleFramesImageAugmentation("val")],
        modality_defaults={
            Modality.LIDAR_PC: LidarPointCloudDefaults(),
            Modality.HD_MAP_BEV: HDMapBEVDefaults.from_adapter(bev_adapter),
            Modality.INTENT: IntentDefaults(),
            Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults(),
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )

    print()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        sensors = batch["current_sensors"]
        labels = batch["current_future_states"]

        cameras = sensors.get_modality_data(Modality.CAMERAS)
        lidar = sensors.get_modality_data(Modality.LIDAR_PC)
        bev = sensors.get_modality_data(Modality.HD_MAP_BEV)
        future = labels.get_modality_data(Modality.FUTURE_STATES)

        per_sample_lidar = torch.bincount(
            lidar.batch_idx, minlength=lidar.batch_size
        ).tolist()
        per_sample_bev_max = bev.amax(dim=(1, 2, 3)).tolist()
        future_xy_shape = future.get(
            [TrajectoryComponent.X, TrajectoryComponent.Y]
        ).shape

        print(f"batch {batch_idx}: datasets={sensors.dataset_name}")
        print(f"  cameras          = {tuple(cameras.shape)}")
        print(f"  lidar pts/sample = {per_sample_lidar}")
        print(
            f"  hdmap_bev        = {tuple(bev.shape)}, "
            f"max/sample = {[f'{m:.2f}' for m in per_sample_bev_max]}"
        )
        print(f"  future_states xy = {tuple(future_xy_shape)}")
        print()


if __name__ == "__main__":
    main()
