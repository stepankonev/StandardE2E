"""CLI to render a single frame from a cached ``.npz`` for visual inspection.

Modalities supported in PR 1: ``cameras``, ``lidar_pc``, ``combined``,
``intent``. PR 2 adds ``hd_map`` and ``detections_3d``.

Conventions:
    * Matplotlib only (no external rendering deps).
    * Spatial views: ego at (0, 0) with +x triangle marker, 1 m grid.
    * 200 DPI PNG; expected output well under 2 MB for a single frame.
    * Title format: ``{dataset}/{segment}/{frame} - {modality}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from standard_e2e.data_structures import (  # noqa: E402
    Detection3D,
    FrameDetections3D,
    HDMapData,
    LidarData,
    Trajectory,
    TransformedFrameData,
)
from standard_e2e.enums import (  # noqa: E402
    CameraDirection,
    Intent,
    Modality,
)
from standard_e2e.enums import TrajectoryComponent as TC  # noqa: E402

# Spatial views (lidar / detections) save at 200 DPI for detail; image-heavy
# views drop to 120 DPI so an 8-cam mosaic stays under the 2 MB cap.
_DPI_SPATIAL = 200
_DPI_IMAGE_HEAVY = 120
_DPI_BY_MODALITY = {
    "cameras": _DPI_IMAGE_HEAVY,
    "combined": _DPI_IMAGE_HEAVY,
    "lidar_pc": _DPI_SPATIAL,
    "intent": _DPI_SPATIAL,
    "hd_map": _DPI_SPATIAL,
    "detections_3d": _DPI_SPATIAL,
}
_BEV_RANGE_M = 60.0  # half-extent for spatial views (+/- 60 m)
_MAX_IMAGE_EDGE_PX = 480  # downsample camera images before plotting

# 8-camera mosaic layout: 3 rows, 3 cols.
_CAMERA_GRID: dict[CameraDirection, tuple[int, int]] = {
    CameraDirection.FRONT_LEFT: (0, 0),
    CameraDirection.FRONT: (0, 1),
    CameraDirection.FRONT_RIGHT: (0, 2),
    CameraDirection.SIDE_LEFT: (1, 0),
    CameraDirection.SIDE_RIGHT: (1, 2),
    CameraDirection.REAR_LEFT: (2, 0),
    CameraDirection.REAR: (2, 1),
    CameraDirection.REAR_RIGHT: (2, 2),
}

_SUPPORTED_MODALITIES = (
    "cameras",
    "lidar_pc",
    "combined",
    "intent",
    "hd_map",
    "detections_3d",
)


def render_frame(
    npz_path: str,
    modality: str,
    out_path: str,
    frame_id: int | None = None,
) -> str:
    """Render one cached frame to a PNG.

    Args:
        npz_path: Path to a ``TransformedFrameData`` ``.npz`` artifact.
        modality: One of ``cameras``, ``lidar_pc``, ``combined``, ``intent``.
        out_path: Destination PNG path.
        frame_id: Optional override for the frame label in the title.

    Returns:
        The output path that was written.

    Raises:
        ValueError: When ``modality`` is not one of the supported names.
    """
    if modality not in _SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported modality {modality!r}; expected one of "
            f"{_SUPPORTED_MODALITIES}"
        )

    frame = TransformedFrameData.from_npz(npz_path)
    label = _title(frame, modality, frame_id)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if modality == "cameras":
        fig = _render_cameras(frame, label)
    elif modality == "lidar_pc":
        fig = _render_lidar_pc(frame, label)
    elif modality == "intent":
        fig = _render_intent(frame, label)
    elif modality == "hd_map":
        fig = _render_hd_map(frame, label)
    elif modality == "detections_3d":
        fig = _render_detections_3d(frame, label)
    else:  # combined
        fig = _render_combined(frame, label)

    fig.savefig(out, dpi=_DPI_BY_MODALITY[modality], bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _title(frame: TransformedFrameData, modality: str, frame_id: int | None) -> str:
    fid = frame.frame_id if frame_id is None else frame_id
    return f"{frame.dataset_name}/{frame.segment_id}/{fid} - {modality}"


def _downsample_image(
    image: np.ndarray, max_edge: int = _MAX_IMAGE_EDGE_PX
) -> np.ndarray:
    """Subsample an HWC uint8 image so its longest edge <= ``max_edge`` pixels."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image
    stride = int(np.ceil(longest / max_edge))
    return image[::stride, ::stride]


def _draw_ego(ax) -> None:
    triangle = np.array([[1.5, 0.0], [-1.0, 0.8], [-1.0, -0.8]])
    ax.fill(triangle[:, 0], triangle[:, 1], color="tab:red", alpha=0.85, zorder=5)
    ax.plot(0, 0, marker="o", color="black", markersize=3, zorder=6)


def _setup_bev(ax, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-_BEV_RANGE_M, _BEV_RANGE_M)
    ax.set_ylim(-_BEV_RANGE_M, _BEV_RANGE_M)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_xticks(np.arange(-_BEV_RANGE_M, _BEV_RANGE_M + 1, 10))
    ax.set_yticks(np.arange(-_BEV_RANGE_M, _BEV_RANGE_M + 1, 10))
    ax.set_xlabel("ego x (m)")
    ax.set_ylabel("ego y (m)")
    ax.set_title(title)


def _render_cameras(frame: TransformedFrameData, title: str):
    cameras = frame.get_modality_data(Modality.CAMERAS)
    if cameras is None or (
        not isinstance(cameras, np.ndarray) and len(cameras) == 0
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no cameras", ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle(title)
        return fig

    # PanoImageAdapter produces a single stitched ndarray; surround mode
    # produces dict[CameraDirection, CameraData].
    if isinstance(cameras, np.ndarray):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(_downsample_image(cameras))
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax in axes.flat:
        ax.set_axis_off()

    for direction, cam in cameras.items():
        if direction not in _CAMERA_GRID:
            continue
        row, col = _CAMERA_GRID[direction]
        ax = axes[row, col]
        ax.imshow(_downsample_image(cam.image))
        ax.set_title(direction.value, fontsize=8)
    # Center cell (1,1) gets ego marker for orientation reference.
    center = axes[1, 1]
    center.set_axis_on()
    _setup_bev(center, "ego")
    _draw_ego(center)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _render_lidar_pc(frame: TransformedFrameData, title: str):
    lidar: LidarData | None = frame.get_modality_data(Modality.LIDAR_PC)
    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_bev(ax, title)
    if lidar is not None and len(lidar.points) > 0:
        x = lidar.points["x"].to_numpy()
        y = lidar.points["y"].to_numpy()
        ax.scatter(x, y, s=0.5, c="tab:blue", alpha=0.6, zorder=2)
    _draw_ego(ax)
    return fig


def _render_intent(frame: TransformedFrameData, title: str):
    intent_value = frame.get_modality_data(Modality.INTENT)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()
    label = "<missing>"
    if intent_value is not None:
        try:
            label = Intent(int(intent_value)).name
        except (ValueError, TypeError):
            label = str(intent_value)
    ax.text(0.5, 0.5, f"INTENT = {label}", ha="center", va="center", fontsize=24)
    fig.suptitle(title)
    return fig


def _render_combined(frame: TransformedFrameData, title: str):
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(14, 7))
    cameras = frame.get_modality_data(Modality.CAMERAS)
    if isinstance(cameras, np.ndarray):
        ax_cam.imshow(_downsample_image(cameras))
        ax_cam.set_title("pano")
    elif cameras is not None and CameraDirection.FRONT in cameras:
        ax_cam.imshow(_downsample_image(cameras[CameraDirection.FRONT].image))
        ax_cam.set_title("FRONT")
    else:
        ax_cam.text(0.5, 0.5, "no FRONT cam", ha="center", va="center")
    ax_cam.set_axis_off()

    _setup_bev(ax_bev, "ego BEV")
    lidar: LidarData | None = frame.get_modality_data(Modality.LIDAR_PC)
    if lidar is not None and len(lidar.points) > 0:
        ax_bev.scatter(
            lidar.points["x"].to_numpy(),
            lidar.points["y"].to_numpy(),
            s=0.5,
            c="tab:blue",
            alpha=0.4,
        )
    _plot_trajectory(ax_bev, frame.get_modality_data(Modality.PAST_STATES), "tab:gray")
    _plot_trajectory(
        ax_bev, frame.get_modality_data(Modality.FUTURE_STATES), "tab:green"
    )
    _draw_ego(ax_bev)

    intent_value = frame.get_modality_data(Modality.INTENT)
    if intent_value is not None:
        try:
            intent_label = Intent(int(intent_value)).name
        except (ValueError, TypeError):
            intent_label = str(intent_value)
        ax_bev.text(
            0.02,
            0.98,
            f"intent: {intent_label}",
            transform=ax_bev.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _render_hd_map(frame: TransformedFrameData, title: str):
    """BEV render of HDMapData (ego frame).

    Lanes are solid lines, lane boundaries dashed, road edges thick
    solid; crosswalks / drivable areas / driveways / speed bumps fill
    polygons; stop signs marker.
    """
    hd_map: HDMapData | None = frame.get_modality_data(Modality.HD_MAP)
    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_bev(ax, title)
    if hd_map is not None:
        for lane in hd_map.lanes:
            xy = lane.centerline[:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", linewidth=1.0, zorder=2)
        for boundary in hd_map.lane_boundaries:
            xy = boundary.polyline[:, :2]
            ax.plot(
                xy[:, 0], xy[:, 1], color="tab:gray", linewidth=0.6,
                linestyle="--", zorder=2,
            )
        for edge in hd_map.road_edges:
            xy = edge.polyline[:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.5, zorder=3)
        for cw in hd_map.crosswalks:
            xy = cw.polygon[:, :2]
            ax.fill(xy[:, 0], xy[:, 1], color="tab:olive", alpha=0.35, zorder=2)
        for da in hd_map.drivable_areas:
            xy = da.polygon[:, :2]
            ax.fill(xy[:, 0], xy[:, 1], color="tab:green", alpha=0.15, zorder=1)
        for ss in hd_map.stop_signs:
            ax.plot(ss.position[0], ss.position[1], marker="s", color="tab:red", markersize=6, zorder=4)
    _draw_ego(ax)
    return fig


def _render_detections_3d(frame: TransformedFrameData, title: str):
    """BEV render of 3D detections as box footprints (ego frame)."""
    detections = frame.get_modality_data(Modality.DETECTIONS_3D)
    fig, ax = plt.subplots(figsize=(8, 8))
    _setup_bev(ax, title)
    iterable: list[Detection3D]
    if isinstance(detections, FrameDetections3D):
        iterable = detections.detections
    elif isinstance(detections, list):
        iterable = [d for d in detections if isinstance(d, Detection3D)]
    else:
        iterable = []
    for det in iterable:
        try:
            xy = det.trajectory.get([TC.X, TC.Y]).reshape(-1, 2)
        except KeyError:
            continue
        if xy.shape[0] == 0:
            continue
        cx, cy = float(xy[0, 0]), float(xy[0, 1])
        # Aggregated trajectories (post FutureDetectionsAggregator) only
        # carry X, Y, HEADING; size components are absent. Fall back to
        # a marker at the center in that case so detections still show.
        try:
            wh = det.trajectory.get([TC.LENGTH, TC.WIDTH]).reshape(-1, 2)
            yaw = float(det.trajectory.get(TC.HEADING).reshape(-1)[0])
            length, width = float(wh[0, 0]), float(wh[0, 1])
        except KeyError:
            ax.plot(cx, cy, marker="x", color="tab:red", markersize=5, zorder=3)
            continue
        corners = _box_corners_2d(cx, cy, length, width, yaw)
        ax.plot(
            corners[[0, 1, 2, 3, 0], 0],
            corners[[0, 1, 2, 3, 0], 1],
            color="tab:red",
            linewidth=1.0,
            zorder=3,
        )
        # Draw the full aggregated trajectory as a thin line for context.
        if xy.shape[0] > 1:
            ax.plot(xy[:, 0], xy[:, 1], color="tab:red", alpha=0.4, linewidth=0.5)
    _draw_ego(ax)
    return fig


def _box_corners_2d(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    half_l = length / 2.0
    half_w = width / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
    rotated = local @ rot.T
    return rotated + np.array([cx, cy], dtype=np.float32)


def _plot_trajectory(ax, traj: Trajectory | None, color: str) -> None:
    if traj is None or traj.length == 0:
        return
    try:
        xy = traj.get([TC.X, TC.Y])
    except KeyError:
        return
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.5, marker=".", markersize=3)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--npz", required=True, help="path to a TransformedFrameData .npz"
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=_SUPPORTED_MODALITIES,
    )
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument(
        "--frame", type=int, default=None, help="optional frame label override"
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    render_frame(
        npz_path=args.npz,
        modality=args.modality,
        out_path=args.out,
        frame_id=args.frame,
    )


if __name__ == "__main__":
    main()
