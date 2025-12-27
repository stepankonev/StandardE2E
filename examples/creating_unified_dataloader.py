"""
This example demonstrates how to create a unified DataLoader for multiple datasets.
Particularly, it focuses on the Waymo End-to-End and Waymo Perception datasets.
"""

import argparse
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from standard_e2e import UnifiedE2EDataset
from standard_e2e.dataset_utils.augmentation import (
    MultipleFramesImageAugmentation,
    TrajectoryResampling,
)
from standard_e2e.dataset_utils.frame_loader import (
    FrameLoader,
)
from standard_e2e.dataset_utils.modality_defaults import IntentDefaults
from standard_e2e.dataset_utils.selector import (
    ClosestTimestampSelector,
    CurrentSelector,
)
from standard_e2e.enums import Intent, Modality
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.indexing import get_multi_dataset_index


def parse_args():
    parser = argparse.ArgumentParser(description="Create a unified DataLoader")
    parser.add_argument(
        "--processed_data_path",
        type=str,
        help="Path to the processed data",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_data_path = args.processed_data_path
    waymo_e2e_train_index = pd.read_parquet(
        os.path.join(processed_data_path, "waymo_e2e/training/index.parquet")
    )
    waymo_perception_train_index = pd.read_parquet(
        os.path.join(processed_data_path, "waymo_perception/training/index.parquet")
    )
    print("Waymo End-to-end dataset size:", waymo_e2e_train_index.shape[0])
    print("Waymo Perception dataset size:", waymo_perception_train_index.shape[0])
    unified_index = get_multi_dataset_index(
        [waymo_e2e_train_index, waymo_perception_train_index]
    )
    feature_loaders = [
        FrameLoader(
            frame_name="current_sensors",
            required_modalities=[
                Modality.CAMERAS,
                Modality.INTENT,
                Modality.PAST_STATES,
            ],
            frame_selector=CurrentSelector(location="features"),
        )
    ]
    label_loaders = [
        FrameLoader(
            frame_name="current_states",
            required_modalities=[
                Modality.FUTURE_STATES,
            ],
            frame_selector=CurrentSelector(location="labels"),
        ),
        FrameLoader(
            frame_name="future_1second",
            required_modalities=[Modality.CAMERAS],
            frame_selector=ClosestTimestampSelector(location="labels", delta_t=1.0),
        ),
    ]

    history_time_lattice = np.linspace(-3, 0, 6, endpoint=False)
    future_time_lattice = np.linspace(0, 10, 21)[1:]

    dataset = UnifiedE2EDataset(
        index_data=unified_index,
        processed_data_path=processed_data_path,
        regime="train",
        feature_loaders=feature_loaders,
        label_loaders=label_loaders,
        augmentations=[
            MultipleFramesImageAugmentation("train"),
            TrajectoryResampling(
                history_target_timestamps=history_time_lattice,
                target_frame_names=["current_sensors"],
            ),
            TrajectoryResampling(
                future_target_timestamps=future_time_lattice,
                target_frame_names=["current_states"],
            ),
        ],
        modality_defaults={Modality.INTENT: IntentDefaults()},
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=16,
    )
    # Further processing with the dataset
    num_e2e, num_perc = 0, 0

    intent_counter = {intent: 0 for intent in Intent}
    pbar = tqdm(dataloader, total=len(dataloader), desc="Processing batches")
    for batch in pbar:
        num_e2e += sum(e == "waymo_e2e" for e in batch["current_sensors"].dataset_name)
        num_perc += sum(
            e == "waymo_perception" for e in batch["current_sensors"].dataset_name
        )
        pbar.set_description(
            f"Waymo E2E samples: {num_e2e}, Waymo Perception samples: {num_perc}"
        )
        for intent in Intent:
            intent_counter[intent] += sum(
                e == intent
                for e in batch["current_sensors"].get_modality_data(Modality.INTENT)
            )
        past_states = (
            batch["current_sensors"]
            .get_modality_data(Modality.PAST_STATES)
            .get([TC.TIMESTAMP, TC.X, TC.Y])
        )
        future_states = (
            batch["current_states"]
            .get_modality_data(Modality.FUTURE_STATES)
            .get([TC.TIMESTAMP, TC.X, TC.Y])
        )
        assert past_states.shape[1:] == (len(history_time_lattice), 3)
        assert future_states.shape[1:] == (len(future_time_lattice), 3)

    print("Intent counts:", intent_counter)


if __name__ == "__main__":
    main()
