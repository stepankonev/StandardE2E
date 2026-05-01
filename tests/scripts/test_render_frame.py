"""Synthetic-fixture tests for ``scripts/render_frame.py``.

PR 1 step 1.1: assert one PNG per modality is produced from a synthetic
``TransformedFrameData``, in the expected size band (>= 5 KB, <= 2 MB).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_e2e.data_structures import (
    CameraData,
    Crosswalk,
    Detection3D,
    DrivableArea,
    FrameDetections3D,
    HDMapData,
    Lane,
    LaneBoundary,
    LidarData,
    RoadEdge,
    StopSign,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import (
    CameraDirection,
    DetectionType,
    Intent,
    LaneMarkType,
    LaneType,
    Modality,
    RoadEdgeType,
    TrajectoryComponent as TC,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_frame.py"
MIN_PNG_BYTES = 5 * 1024
MAX_PNG_BYTES = 2 * 1024 * 1024


@pytest.fixture(scope="module")
def render_module():
    spec = importlib.util.spec_from_file_location("_render_frame", RENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load render script from {RENDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_render_frame"] = module
    spec.loader.exec_module(module)
    return module


def _make_camera(direction: CameraDirection, h: int = 32, w: int = 48) -> CameraData:
    rng = np.random.default_rng(int(direction.value.encode().hex(), 16) % (2**32))
    image = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return CameraData(
        camera_direction=direction,
        image=image,
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
    )


def _make_cameras_8() -> dict[CameraDirection, CameraData]:
    return {direction: _make_camera(direction) for direction in CameraDirection}


def _make_lidar(n_points: int = 256) -> LidarData:
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=10.0, size=(n_points, 3)).astype(np.float32)
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})
    return LidarData(points=df)


def _make_trajectory(start: float, stop: float, n: int = 5) -> Trajectory:
    xs = np.linspace(start, stop, n).astype(np.float32)
    ys = np.zeros(n, dtype=np.float32)
    ts = np.linspace(start, stop, n).astype(np.float32) * 0.25
    return Trajectory(
        data={TC.X: xs, TC.Y: ys, TC.Z: np.zeros(n, dtype=np.float32), TC.TIMESTAMP: ts}
    )


def _make_hd_map() -> HDMapData:
    return HDMapData(
        lanes=[
            Lane(
                centerline=np.array([[-20, 0, 0], [-10, 0, 0], [0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32),
                lane_type=LaneType.VEHICLE,
            )
        ],
        lane_boundaries=[
            LaneBoundary(
                polyline=np.array([[-20, 1.5, 0], [20, 1.5, 0]], dtype=np.float32),
                boundary_type=LaneMarkType.SOLID_WHITE,
            )
        ],
        road_edges=[
            RoadEdge(
                polyline=np.array([[-20, 4, 0], [20, 4, 0]], dtype=np.float32),
                road_edge_type=RoadEdgeType.BOUNDARY,
            )
        ],
        crosswalks=[
            Crosswalk(polygon=np.array([[5, -2, 0], [10, -2, 0], [10, 2, 0], [5, 2, 0]], dtype=np.float32))
        ],
        stop_signs=[StopSign(position=np.array([15, 1, 0], dtype=np.float32))],
        drivable_areas=[
            DrivableArea(polygon=np.array([[-30, -5, 0], [30, -5, 0], [30, 5, 0], [-30, 5, 0]], dtype=np.float32))
        ],
    )


def _make_frame_detections() -> FrameDetections3D:
    det = Detection3D(
        unique_agent_id="a-0",
        detection_type=DetectionType.VEHICLE,
        trajectory=Trajectory(
            data={
                TC.X: np.array([10.0], dtype=np.float32),
                TC.Y: np.array([2.0], dtype=np.float32),
                TC.Z: np.array([0.0], dtype=np.float32),
                TC.HEADING: np.array([0.0], dtype=np.float32),
                TC.LENGTH: np.array([4.5], dtype=np.float32),
                TC.WIDTH: np.array([2.0], dtype=np.float32),
                TC.HEIGHT: np.array([1.5], dtype=np.float32),
                TC.TIMESTAMP: np.array([0.0], dtype=np.float32),
            }
        ),
    )
    return FrameDetections3D(detections=[det])


def _build_synthetic_frame() -> TransformedFrameData:
    frame = TransformedFrameData(
        dataset_name="synthetic",
        segment_id="seg0",
        frame_id=42,
        timestamp=1.234,
        split="test",
    )
    frame.set_modality_data(Modality.CAMERAS, _make_cameras_8())
    frame.set_modality_data(Modality.LIDAR_PC, _make_lidar())
    frame.set_modality_data(Modality.INTENT, Intent.GO_LEFT)
    frame.set_modality_data(Modality.PAST_STATES, _make_trajectory(-4, 0))
    frame.set_modality_data(Modality.FUTURE_STATES, _make_trajectory(0, 8))
    frame.set_modality_data(Modality.HD_MAP, _make_hd_map())
    frame.set_modality_data(Modality.DETECTIONS_3D, _make_frame_detections())
    return frame


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    frame = _build_synthetic_frame()
    out = tmp_path / "frame_synth.npz"
    frame.to_npz(str(out))
    return out


@pytest.mark.parametrize(
    "modality", ["cameras", "lidar_pc", "combined", "intent", "hd_map", "detections_3d"]
)
def test_render_modality_produces_png(
    render_module, tmp_path: Path, synthetic_npz: Path, modality: str
):
    out = tmp_path / f"{modality}.png"
    render_module.render_frame(
        npz_path=str(synthetic_npz),
        modality=modality,
        out_path=str(out),
    )
    assert out.exists(), f"render_frame did not produce {out}"
    size = out.stat().st_size
    assert size >= MIN_PNG_BYTES, f"PNG suspiciously small: {size} bytes"
    assert size <= MAX_PNG_BYTES, f"PNG exceeds 2 MB cap: {size} bytes"
    # PNG magic
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_unknown_modality_raises(
    render_module, tmp_path: Path, synthetic_npz: Path
):
    out = tmp_path / "out.png"
    with pytest.raises((ValueError, KeyError)):
        render_module.render_frame(
            npz_path=str(synthetic_npz),
            modality="not_a_modality",
            out_path=str(out),
        )


def test_render_cli_invocation(tmp_path: Path, synthetic_npz: Path):
    """Runs the script via subprocess to exercise the CLI surface."""
    import subprocess

    out = tmp_path / "cli_cameras.png"
    result = subprocess.run(
        [
            sys.executable,
            str(RENDER_SCRIPT),
            "--npz",
            str(synthetic_npz),
            "--modality",
            "cameras",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert out.stat().st_size >= MIN_PNG_BYTES
