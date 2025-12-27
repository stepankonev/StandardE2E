import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from standard_e2e import UnifiedE2EDataset
from standard_e2e.dataset_utils.augmentation import MultipleFramesImageAugmentation
from standard_e2e.dataset_utils.modality_defaults import (
    ModalityDefaults,
    PreferredTrajectoryDefaults,
)
from standard_e2e.enums import Modality, TrajectoryComponent
from standard_e2e.indexing.filters import FrameFilterByBooleanColumn, IndexFilter
from standard_e2e.utils import load_yaml_config

XY_COMPONENTS = [TrajectoryComponent.X, TrajectoryComponent.Y]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_path",
        type=str,
        help="Path to the processed data",
        required=True,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration file",
        default="configs/base.yaml",
    )
    return parser.parse_args()


def fix_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VerySimpleE2EModel(nn.Module):
    """A very simple end-to-end model for demonstration purposes."""

    def __init__(self, out_dim: int = 40):
        super(VerySimpleE2EModel, self).__init__()
        self._cnn_backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self._cnn_backbone.fc = nn.Identity()
        self._out_dim = out_dim
        self._fusion_layer = nn.Sequential(
            nn.BatchNorm1d(548),
            nn.ReLU(),
            nn.Linear(548, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(
        self, image: torch.Tensor, intent: torch.Tensor, past_states: torch.Tensor
    ) -> torch.Tensor:
        bs = image.size(0)
        image_embedding = self._cnn_backbone(image)
        # pylint: disable=not-callable
        intent_ohe = torch.nn.functional.one_hot(intent.long(), num_classes=4).float()
        past_state_embedding = past_states.view(bs, -1)
        x = torch.cat([image_embedding, intent_ohe, past_state_embedding], dim=1)
        x = self._fusion_layer(x)
        x = x.view(bs, -1, 2)
        return x


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        loss = loss * mask
        return loss.mean()


def create_dataloader(
    config: dict,
    processed_data_path: str,
    index_df: pd.DataFrame,
    regime: str,
    batch_size: int = 64,
    num_workers: int = 1,
    index_filters: list[IndexFilter] | None = None,
    modality_defaults: dict[Modality, ModalityDefaults] | None = None,
):
    """
    Create a DataLoader for the UnifiedE2EDataset.
    """
    dataset = UnifiedE2EDataset(
        processed_data_path=processed_data_path,
        feature_loaders_config=config["dataset"]["features"],
        label_loaders_config=config["dataset"]["labels"],
        index_data=index_df,
        regime=regime,
        augmentations=[MultipleFramesImageAugmentation(regime)],
        index_filters=index_filters,
        modality_defaults=modality_defaults,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=regime == "train",
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
    )
    return dataloader


def main():
    fix_seeds(42)  # Set random seeds for reproducibility
    args = parse_args()

    processed_data_path = args.processed_data_path
    train_index_df_path = os.path.join(
        processed_data_path, "waymo_e2e/training/index.parquet"
    )
    val_index_df_path = os.path.join(processed_data_path, "waymo_e2e/val/index.parquet")
    config_path = args.config_file
    config = load_yaml_config(config_path)
    train_index_data = pd.read_parquet(train_index_df_path)
    val_index_data = pd.read_parquet(val_index_df_path)
    train_dataloader = create_dataloader(
        config=config,
        processed_data_path=processed_data_path,
        index_df=train_index_data,
        regime="train",
        modality_defaults={
            Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults()
        },
    )
    val_dataloader = create_dataloader(
        config=config,
        processed_data_path=processed_data_path,
        index_df=val_index_data,
        regime="val",
        modality_defaults={
            Modality.PREFERENCE_TRAJECTORY: PreferredTrajectoryDefaults()
        },
    )
    filter_only_with_preference_trajectory = FrameFilterByBooleanColumn(
        boolean_column="extra_has_preference_trajectories"
    )
    val_dataloader_with_filter = create_dataloader(
        config=config,
        processed_data_path=processed_data_path,
        index_df=val_index_data,
        regime="val",
        index_filters=[filter_only_with_preference_trajectory],
    )
    model = VerySimpleE2EModel()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=0.0003)
    criterion = torch.nn.MSELoss()
    masked_criterion = MaskedMSELoss()
    # Further processing with the dataset
    for epoch in range(20):
        train_loss = []
        val_loss = []
        preferred_trajectory_loss = []
        model.train()
        pbar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Training Epoch {epoch + 1}",
        )
        for batch in pbar:
            batch["current_sensors"] = batch["current_sensors"].cuda()
            optimizer.zero_grad()
            outputs = model(
                batch["current_sensors"].get_modality_data(Modality.CAMERAS),
                batch["current_sensors"].get_modality_data(Modality.INTENT),
                batch["current_sensors"]
                .get_modality_data(Modality.PAST_STATES)
                .get(XY_COMPONENTS),
            )
            loss = criterion(
                outputs,
                batch["current_future_states"]
                .get_modality_data(Modality.FUTURE_STATES)
                .get(XY_COMPONENTS)
                .cuda(),
            )
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"Training Epoch {epoch + 1}, loss: {np.mean(train_loss[-100:]):.4f}"
            )

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}"):
                batch["current_sensors"] = batch["current_sensors"].cuda()
                outputs = model(
                    batch["current_sensors"].get_modality_data(Modality.CAMERAS),
                    batch["current_sensors"].get_modality_data(Modality.INTENT),
                    batch["current_sensors"]
                    .get_modality_data(Modality.PAST_STATES)
                    .get(XY_COMPONENTS),
                )
                loss = criterion(
                    outputs,
                    batch["current_future_states"]
                    .get_modality_data(Modality.FUTURE_STATES)
                    .get(XY_COMPONENTS)
                    .cuda(),
                )
                val_loss.append(loss.item())
            for batch in tqdm(
                val_dataloader_with_filter,
                desc=f"Filtered Validation Epoch {epoch + 1}",
            ):
                batch["current_sensors"] = batch["current_sensors"].cuda()
                outputs = model(
                    batch["current_sensors"].get_modality_data(Modality.CAMERAS),
                    batch["current_sensors"].get_modality_data(Modality.INTENT),
                    batch["current_sensors"]
                    .get_modality_data(Modality.PAST_STATES)
                    .get(XY_COMPONENTS),
                )
                loss = masked_criterion(
                    outputs,
                    batch["current_future_states"]
                    .get_modality_data(Modality.PREFERENCE_TRAJECTORY)[0]
                    .get(XY_COMPONENTS)[:, 1:]
                    .cuda(),
                    batch["current_future_states"]
                    .get_modality_data(Modality.PREFERENCE_TRAJECTORY)[0]
                    .get(TrajectoryComponent.IS_VALID)[:, 1:]
                    .cuda(),
                )
                preferred_trajectory_loss.append(loss.item())
        print(
            f"Epoch {epoch + 1} - Train Loss: {np.mean(train_loss):.4f}, "
            f"Val Loss: {np.mean(val_loss):.4f}, "
            f"Preferred Trajectory Loss: {np.mean(preferred_trajectory_loss):.4f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
