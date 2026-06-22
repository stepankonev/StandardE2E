# flake8: noqa: E501
"""Tests for the universal processed-output visualization tool.

Covers the converter's ``dataset_info.yaml`` emission, the BEV adapters' grid
metadata, the per-frame renderer's modality auto-detection across diverse
modality combinations (dict cameras, pano cameras, rasters, point clouds, vector
detections, trajectories, and a near-empty frame), scene selection, and the CLI
end-to-end (synthetic npz + index -> mp4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from standard_e2e.caching.adapters import HDMapBEVAdapter, LidarAdapter
from standard_e2e.caching.source_dataset_converter import SourceDatasetConverter
from standard_e2e.constants import DATASET_INFO_FILE_NAME, INDEX_FILE_NAME
from standard_e2e.data_structures import (
    CameraData,
    Detection3D,
    FrameDetections3D,
    LidarPointCloud,
    Trajectory,
)
from standard_e2e.data_structures.frame_data import TransformedFrameData
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    LidarComponent,
    MapElementType,
    Modality,
)
from standard_e2e.enums import TrajectoryComponent as TC
from standard_e2e.visualization import render_frame
from standard_e2e.visualization.render import figure_to_bgr
from standard_e2e.visualization.visualize_processed import _infer_fps, _select_segments
from standard_e2e.visualization.visualize_processed import main as visualize_main


# --------------------------------------------------------------------------- #
# dataset_info.yaml emission (converter) + adapter.spec
# --------------------------------------------------------------------------- #
class _StubProcessor:
    dataset_name = "stub_dataset"
    split = "val"

    def __init__(self, output_path: str):
        self.specific_output_path = output_path
        self.adapters = [LidarAdapter(), HDMapBEVAdapter(min_x=-16.0, max_x=16.0)]


class _StubConverter(SourceDatasetConverter):
    def _get_source_dataset_iterator(self):
        return iter([])


def test_converter_writes_dataset_info_yaml(tmp_path):
    proc = _StubProcessor(str(tmp_path))
    conv = _StubConverter(source_processor=proc, input_path=".", split="val")
    conv._write_dataset_info()

    info = yaml.safe_load((tmp_path / DATASET_INFO_FILE_NAME).read_text())
    assert info["dataset_name"] == "stub_dataset"
    assert info["split"] == "val"
    names = [a["name"] for a in info["adapters"]]
    assert names == ["LidarAdapter", "HDMapBEVAdapter"]
    hdmap = info["adapters"][1]
    assert hdmap["metadata"]["hd_map_bev_grid"]["min_x"] == -16.0
    assert hdmap["metadata"]["hd_map_bev_channels"][0] == "lane_center"


def test_adapter_spec_shape():
    spec = HDMapBEVAdapter().spec
    assert spec["name"] == "HDMapBEVAdapter"
    assert "hd_map_bev_grid" in spec["metadata"]


# --------------------------------------------------------------------------- #
# Synthetic frame builders
# --------------------------------------------------------------------------- #
def _camera(direction: CameraDirection) -> CameraData:
    return CameraData(
        camera_direction=direction,
        image=np.zeros((8, 12, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
    )


def _raster(channels: int, h: int = 8, w: int = 8) -> np.ndarray:
    arr = np.zeros((channels, h, w), dtype=np.float32)
    arr[0, 2:5, 2:5] = 1.0
    return arr


_GRID = {
    "min_x": -16.0,
    "max_x": 16.0,
    "min_y": -16.0,
    "max_y": 16.0,
    "pixels_per_meter": 0.25,
}


def _traj(n: int = 5) -> Trajectory:
    return Trajectory(
        {
            TC.TIMESTAMP: list(np.arange(n, dtype=float)),
            TC.X: list(np.linspace(0, 10, n)),
            TC.Y: list(np.zeros(n)),
        }
    )


def _detections() -> FrameDetections3D:
    det = Detection3D(
        unique_agent_id="a0",
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            {
                TC.TIMESTAMP: [0.0],
                TC.X: [6.0],
                TC.Y: [2.0],
                TC.HEADING: [0.3],
                TC.LENGTH: [4.0],
                TC.WIDTH: [1.8],
                TC.HEIGHT: [1.5],
            }
        ),
    )
    return FrameDetections3D(detections=[det])


def _frame(
    modalities: dict,
    aux: dict | None = None,
    frame_id: int = 0,
    segment_id: str = "seg0",
) -> TransformedFrameData:
    fd = TransformedFrameData(
        dataset_name="d",
        segment_id=segment_id,
        frame_id=frame_id,
        timestamp=0.1 * frame_id,
        split="val",
        aux_data=aux or {},
    )
    for modality, data in modalities.items():
        fd.set_modality_data(modality, data)
    return fd


# --------------------------------------------------------------------------- #
# Renderer auto-detects diverse modality combinations
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fig():
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 4))
    yield figure
    plt.close(figure)


def _assert_renders(fig, frame):
    render_frame(fig, frame)
    out = figure_to_bgr(fig)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_render_dict_cameras_plus_map_lidar_traj(fig):
    frame = _frame(
        {
            Modality.CAMERAS: {
                CameraDirection.FRONT: _camera(CameraDirection.FRONT),
                CameraDirection.REAR: _camera(CameraDirection.REAR),
            },
            Modality.HD_MAP_BEV: _raster(len(MapElementType)),
            Modality.LIDAR_PC: LidarPointCloud(
                np.array([[1, 0, 0], [2, 1, 0.5]], dtype=np.float32),
                [LidarComponent.X, LidarComponent.Y, LidarComponent.Z],
            ),
            Modality.PAST_STATES: _traj(),
            Modality.FUTURE_STATES: _traj(),
        },
        aux={
            "hd_map_bev_grid": _GRID,
            "hd_map_bev_channels": [t.value for t in MapElementType],
        },
    )
    _assert_renders(fig, frame)


def test_render_pano_cameras_and_detections(fig):
    frame = _frame(
        {
            Modality.CAMERAS: np.zeros((20, 60, 3), dtype=np.uint8),  # pano ndarray
            Modality.DETECTIONS_3D: _detections(),
        }
    )
    _assert_renders(fig, frame)


def test_render_lidar_bev_and_detections_bev(fig):
    frame = _frame(
        {
            Modality.LIDAR_BEV: _raster(1),
            Modality.DETECTIONS_3D_BEV: _raster(len(DetectionType)),
        },
        aux={
            "lidar_bev_grid": _GRID,
            "lidar_bev_channels": ["above"],
            "detections_3d_bev_grid": _GRID,
            "detections_3d_bev_channels": [t.value for t in DetectionType],
        },
    )
    _assert_renders(fig, frame)


def test_render_minimal_trajectory_only(fig):
    _assert_renders(fig, _frame({Modality.FUTURE_STATES: _traj()}))


# --------------------------------------------------------------------------- #
# Scene selection
# --------------------------------------------------------------------------- #
def _index(segment_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"segment_id": segment_ids})


def test_select_segments_default_first_one():
    idx = _index(["a", "a", "b", "c"])
    assert _select_segments(idx, None, None) == ["a"]


def test_select_segments_num_scenes():
    idx = _index(["a", "b", "c"])
    assert _select_segments(idx, None, 2) == ["a", "b"]


def test_select_segments_by_id():
    idx = _index(["a", "b", "c"])
    assert _select_segments(idx, ["c", "a"], None) == ["c", "a"]


def test_select_segments_unknown_id_raises():
    with pytest.raises(ValueError):
        _select_segments(_index(["a", "b"]), ["zzz"], None)


# --------------------------------------------------------------------------- #
# fps inference (real-time playback)
# --------------------------------------------------------------------------- #
def test_infer_fps_from_timestamps():
    assert _infer_fps(np.array([0.0, 0.1, 0.2, 0.3])) == pytest.approx(10.0)  # 10 Hz
    assert _infer_fps(np.array([0.0, 0.5, 1.0, 1.5])) == pytest.approx(2.0)  # 2 Hz
    # Robust to ordering and an outlier gap (uses the median interval).
    assert _infer_fps(np.array([0.3, 0.0, 0.1, 0.2, 5.0])) == pytest.approx(10.0)


def test_infer_fps_fallback_and_clamp():
    assert _infer_fps(np.array([1.0])) == 10.0  # too few -> default
    assert _infer_fps(np.array([2.0, 2.0])) == 10.0  # zero interval -> default
    assert _infer_fps(np.array([0.0, 0.001])) == 60.0  # 1000 Hz clamped to 60


# --------------------------------------------------------------------------- #
# CLI end-to-end (synthetic npz + index -> mp4)
# --------------------------------------------------------------------------- #
def test_cli_renders_video_from_processed_folder(tmp_path):
    folder = tmp_path / "d" / "val"
    folder.mkdir(parents=True)

    rows = []
    for frame_id in range(3):
        frame = _frame(
            {
                Modality.CAMERAS: {
                    CameraDirection.FRONT: _camera(CameraDirection.FRONT)
                },
                Modality.LIDAR_PC: LidarPointCloud(
                    np.array([[1, 0, 0], [2, 1, 0]], dtype=np.float32),
                    [LidarComponent.X, LidarComponent.Y, LidarComponent.Z],
                ),
                Modality.FUTURE_STATES: _traj(),
            },
            frame_id=frame_id,
        )
        name = f"seg0_{frame_id}.npz"
        frame.to_npz(str(folder / name))
        rows.append(
            {
                "dataset_name": "d",
                "segment_id": "seg0",
                "frame_id": frame_id,
                "timestamp": 0.1 * frame_id,
                "split": "val",
                "filename": f"d/val/{name}",
            }
        )
    pd.DataFrame(rows).to_parquet(folder / INDEX_FILE_NAME, index=False)
    (folder / DATASET_INFO_FILE_NAME).write_text(
        yaml.safe_dump({"dataset_name": "d", "split": "val", "adapters": []})
    )

    out_dir = tmp_path / "viz"
    visualize_main(
        [str(folder), "--out", str(out_dir), "--num-scenes", "1", "--fps", "5"]
    )

    videos = list(out_dir.glob("*.mp4"))
    assert len(videos) == 1
    assert videos[0].stat().st_size > 0
