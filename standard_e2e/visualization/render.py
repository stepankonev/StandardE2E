"""Universal per-frame rendering for processed ``TransformedFrameData``.

The renderer **auto-detects** whichever modalities a frame carries (it does not
assume any particular dataset) and lays them out as:

* a **camera** mosaic (left) -- a ``dict[CameraDirection, CameraData]`` is shown
  as a surround grid; a single stitched panorama (a bare ndarray, as the pano
  adapter emits) is shown as one wide panel;
* a single **BEV** panel (right) onto which every bird's-eye modality is
  co-registered in meters: the raster BEVs (``hd_map_bev`` / ``lidar_bev`` /
  ``detections_3d_bev``), the ``lidar_pc`` point cloud, vector ``detections_3d``
  boxes, and the ``past_states`` / ``future_states`` / ``preference_trajectory``
  ego trajectories, plus the ego marker.

BEV meter extents and channel orders are read from each frame's ``aux_data``
(written there by the BEV adapters' ``metadata``), so the panel is correct for
any grid configuration without hard-coding it.

Axis convention (matches the BEV adapters): vehicle **x is forward** (up) and
**y is left** (left).
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from standard_e2e.data_structures.frame_data import TransformedFrameData  # noqa: E402
from standard_e2e.enums import CameraDirection, Modality  # noqa: E402
from standard_e2e.enums import TrajectoryComponent as TC  # noqa: E402

# Map element-type -> RGB and detection-type -> RGB palettes (by enum *value*, so
# they apply across datasets). Unknown channel names fall back to grey.
_MAP_COLORS = {
    "lane_center": (0.20, 0.55, 0.95),
    "lane_boundary": (0.95, 0.85, 0.20),
    "road_edge": (0.55, 0.55, 0.55),
    "crosswalk": (0.95, 0.40, 0.70),
    "intersection": (0.55, 0.35, 0.80),
    "drivable_area": (0.15, 0.45, 0.25),
    "stop_line": (0.95, 0.25, 0.25),
    "stop_sign": (0.80, 0.10, 0.10),
    "speed_bump": (0.95, 0.60, 0.10),
    "driveway": (0.50, 0.40, 0.25),
    "walkway": (0.40, 0.75, 0.75),
    "traffic_light": (0.95, 0.60, 0.10),
    "unknown": (0.60, 0.60, 0.60),
}
_DETECTION_COLORS = {
    "vehicle": (0.20, 0.60, 1.00),
    "pedestrian": (1.00, 0.40, 0.40),
    "bicycle": (0.30, 0.95, 0.55),
    "sign": (1.00, 0.75, 0.15),
    "unknown": (0.75, 0.75, 0.75),
}
_FALLBACK_COLOR = (0.60, 0.60, 0.60)

# Camera directions in a stable, spatially-sensible order; present cameras are
# laid out row-major over 3 columns following this order (front row, then sides,
# then rear, then any rig-specific extras).
_CAMERA_ORDER = [
    CameraDirection.FRONT_LEFT,
    CameraDirection.FRONT,
    CameraDirection.FRONT_RIGHT,
    CameraDirection.FRONT_LEFT_NARROW,
    CameraDirection.FRONT_RIGHT_NARROW,
    CameraDirection.SIDE_LEFT,
    CameraDirection.SIDE_RIGHT,
    CameraDirection.SIDE_LEFT_BACK,
    CameraDirection.SIDE_RIGHT_BACK,
    CameraDirection.REAR_LEFT,
    CameraDirection.REAR,
    CameraDirection.REAR_RIGHT,
]

# BEV raster modalities and the aux_data key prefixes they expose.
_RASTER_BEV_MODALITIES = (
    Modality.HD_MAP_BEV,
    Modality.DETECTIONS_3D_BEV,
    Modality.LIDAR_BEV,
)
_TRAJECTORY_MODALITIES = (
    (Modality.PAST_STATES, "cyan", "past"),
    (Modality.FUTURE_STATES, "magenta", "future"),
    (Modality.PREFERENCE_TRAJECTORY, "yellow", "preference"),
)
_DEFAULT_BEV_HALF_EXTENT_M = 50.0
_MAX_LIDAR_POINTS = 30000

CAMERA_PANE_WIDTH_PX = 480


# --------------------------------------------------------------------------- #
# Modality access helpers
# --------------------------------------------------------------------------- #
def _modality(frame: TransformedFrameData, modality: Modality) -> Any:
    """Raw modality payload (no default normalization), or None if absent."""
    if modality not in frame.get_present_modality_keys():
        return None
    return frame.get_modality_data(modality, set_default=False)


def _aux(frame: TransformedFrameData) -> dict[str, Any]:
    return frame.aux_data or {}


def _bev_grid(frame: TransformedFrameData, modality: Modality) -> Optional[dict]:
    """The ``{min_x,max_x,min_y,max_y,pixels_per_meter}`` grid an adapter recorded
    in ``aux_data`` for a raster BEV modality, or None."""
    grid = _aux(frame).get(f"{modality.value}_grid")
    return grid if isinstance(grid, dict) else None


def _bev_channels(frame: TransformedFrameData, modality: Modality) -> Optional[list]:
    channels = _aux(frame).get(f"{modality.value}_channels")
    return list(channels) if channels is not None else None


# --------------------------------------------------------------------------- #
# Camera mosaic
# --------------------------------------------------------------------------- #
def _present_camera_directions(cameras: dict) -> list[CameraDirection]:
    ordered = [d for d in _CAMERA_ORDER if d in cameras]
    extras = [d for d in cameras if d not in _CAMERA_ORDER]
    return ordered + extras


def _render_cameras(fig: Figure, subplotspec, cameras: Any) -> None:
    """Render the camera modality into ``subplotspec`` (dict grid or pano image)."""
    # Pano adapter emits a single stitched ndarray rather than a per-camera dict.
    if isinstance(cameras, np.ndarray):
        ax = fig.add_subplot(subplotspec)
        ax.imshow(cameras)
        ax.set_title("cameras (panorama)", fontsize=9, color="white")
        ax.axis("off")
        return
    if not isinstance(cameras, dict) or not cameras:
        return
    directions = _present_camera_directions(cameras)
    ncols = min(3, len(directions))
    nrows = int(np.ceil(len(directions) / ncols))
    grid = GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=subplotspec, wspace=0.02, hspace=0.12
    )
    for idx, direction in enumerate(directions):
        ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
        camera = cameras[direction]
        image = camera.image if hasattr(camera, "image") else camera
        ax.imshow(np.asarray(image))
        ax.set_title(direction.value, fontsize=9, color="white")
        ax.axis("off")


# --------------------------------------------------------------------------- #
# BEV panel
# --------------------------------------------------------------------------- #
def _bev_extent(frame: TransformedFrameData) -> tuple[float, float, float, float]:
    """``(min_x, max_x, min_y, max_y)`` for the BEV axis: the union of the raster
    grids if any, else a default square around the ego."""
    grids: list[dict] = []
    for modality in _RASTER_BEV_MODALITIES:
        grid = _bev_grid(frame, modality)
        if grid is not None:
            grids.append(grid)
    if grids:
        return (
            min(g["min_x"] for g in grids),
            max(g["max_x"] for g in grids),
            min(g["min_y"] for g in grids),
            max(g["max_y"] for g in grids),
        )
    h = _DEFAULT_BEV_HALF_EXTENT_M
    return (-h, h, -h, h)


def _raster_to_rgba(
    raster: np.ndarray, channels: Optional[list], palette: dict, alpha: float
) -> np.ndarray:
    """Composite a ``(C, H, W)`` class raster to ``(H, W, 4)`` RGBA, coloring each
    channel by its name (earlier channels painted last, i.e. on top)."""
    _, h, w = raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for channel in reversed(range(raster.shape[0])):
        name = channels[channel] if channels and channel < len(channels) else None
        color = palette.get(name, _FALLBACK_COLOR)
        mask = raster[channel] > 0.5
        if mask.any():
            rgba[mask, :3] = color
            rgba[mask, 3] = alpha
    return rgba


def _draw_raster(ax, frame, modality, grid, palette, alpha, zorder) -> None:
    raster = np.asarray(_modality(frame, modality))
    if raster.ndim != 3:
        return
    extent = (grid["min_y"], grid["max_y"], grid["min_x"], grid["max_x"])
    if palette is None:  # density raster (lidar_bev): show intensity
        intensity = raster.max(axis=0)
        rgba = plt.get_cmap("cividis")(np.clip(intensity, 0.0, 1.0))
        rgba[..., 3] = (intensity > 0).astype(np.float32) * alpha
    else:
        rgba = _raster_to_rgba(raster, _bev_channels(frame, modality), palette, alpha)
    ax.imshow(
        rgba, extent=extent, origin="lower", zorder=zorder, interpolation="nearest"
    )


def _draw_lidar_pc(ax, frame, extent, zorder) -> None:
    cloud = _modality(frame, Modality.LIDAR_PC)
    if cloud is None:
        return
    pts = np.asarray(cloud.points if hasattr(cloud, "points") else cloud)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    min_x, max_x, min_y, max_y = extent
    keep = (
        (pts[:, 0] > min_x)
        & (pts[:, 0] < max_x)
        & (pts[:, 1] > min_y)
        & (pts[:, 1] < max_y)
    )
    pts = pts[keep]
    if len(pts) > _MAX_LIDAR_POINTS:
        idx = np.linspace(0, len(pts) - 1, _MAX_LIDAR_POINTS).astype(np.int64)
        pts = pts[idx]
    color = pts[:, 2] if pts.shape[1] >= 3 else "lightgrey"
    ax.scatter(
        pts[:, 1],
        pts[:, 0],
        c=color,
        s=1.0,
        cmap="viridis",
        alpha=0.55,
        zorder=zorder,
        linewidths=0,
    )


def _draw_detection_boxes(ax, frame, zorder) -> None:
    detections = _modality(frame, Modality.DETECTIONS_3D)
    if detections is None or not getattr(detections, "detections", None):
        return
    for det in detections.detections:
        box = det.trajectory.get(
            [TC.X, TC.Y, TC.HEADING, TC.LENGTH, TC.WIDTH], strict=False
        )
        if box is None or len(box) == 0:
            continue
        cx, cy, heading, length, width = (float(v) for v in box[0])
        if length <= 0.0 or width <= 0.0:
            continue
        half = np.array(
            [
                [0.5 * length, 0.5 * width],
                [0.5 * length, -0.5 * width],
                [-0.5 * length, -0.5 * width],
                [-0.5 * length, 0.5 * width],
            ]
        )
        rot = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        corners = half @ rot.T + np.array([cx, cy])
        color = _DETECTION_COLORS.get(det.detection_type.value, _FALLBACK_COLOR)
        ax.add_patch(
            Polygon(
                np.column_stack([corners[:, 1], corners[:, 0]]),
                closed=True,
                fill=False,
                edgecolor=color,
                lw=1.3,
                zorder=zorder,
            )
        )


def _draw_trajectories(ax, frame, zorder) -> None:
    for modality, color, label in _TRAJECTORY_MODALITIES:
        traj = _modality(frame, modality)
        if traj is None:
            continue
        # past/future are a single Trajectory; preference_trajectory is a list of
        # them (e.g. the LongTail counterfactuals). Normalise and draw each, with
        # a single legend entry per modality.
        trajectories = list(traj) if isinstance(traj, (list, tuple)) else [traj]
        style = "--" if modality == Modality.PREFERENCE_TRAJECTORY else "-"
        labelled = False
        for one in trajectories:
            x = np.asarray(one.get(TC.X)).reshape(-1)
            y = np.asarray(one.get(TC.Y)).reshape(-1)
            if x.size == 0:
                continue
            ax.plot(
                y,
                x,
                style,
                color=color,
                lw=2.2,
                zorder=zorder,
                label=None if labelled else label,
            )
            labelled = True


def _draw_bev(ax, frame: TransformedFrameData) -> bool:
    """Draw every present BEV modality onto ``ax``. Returns True if anything was
    drawn (besides the ego marker)."""
    extent = _bev_extent(frame)
    min_x, max_x, min_y, max_y = extent
    ax.set_facecolor("black")

    drew = False
    # Rasters first (background), then point cloud, boxes, trajectories on top.
    if _bev_grid(frame, Modality.HD_MAP_BEV):
        _draw_raster(
            ax,
            frame,
            Modality.HD_MAP_BEV,
            _bev_grid(frame, Modality.HD_MAP_BEV),
            _MAP_COLORS,
            alpha=0.85,
            zorder=2,
        )
        drew = True
    if _bev_grid(frame, Modality.LIDAR_BEV):
        _draw_raster(
            ax,
            frame,
            Modality.LIDAR_BEV,
            _bev_grid(frame, Modality.LIDAR_BEV),
            None,
            alpha=0.8,
            zorder=3,
        )
        drew = True
    if _modality(frame, Modality.LIDAR_PC) is not None:
        _draw_lidar_pc(ax, frame, extent, zorder=3)
        drew = True
    if _bev_grid(frame, Modality.DETECTIONS_3D_BEV):
        _draw_raster(
            ax,
            frame,
            Modality.DETECTIONS_3D_BEV,
            _bev_grid(frame, Modality.DETECTIONS_3D_BEV),
            _DETECTION_COLORS,
            alpha=0.7,
            zorder=4,
        )
        drew = True
    if _modality(frame, Modality.DETECTIONS_3D) is not None:
        _draw_detection_boxes(ax, frame, zorder=5)
        drew = True
    _draw_trajectories(ax, frame, zorder=6)

    # Ego marker + forward arrow at the origin.
    ax.plot(0, 0, "^", color="red", ms=9, zorder=7)
    ax.arrow(
        0,
        0,
        0,
        0.08 * (max_x - min_x),
        color="red",
        width=0.4,
        head_width=2.0,
        length_includes_head=True,
        zorder=7,
    )

    ax.set_xlim(max_y, min_y)  # +y (left) on the left
    ax.set_ylim(min_x, max_x)  # +x (forward) up
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BEV (ego-centric: x forward / y left)", fontsize=9, color="white")
    handles = ax.get_legend_handles_labels()[0]
    if handles:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    return drew


# --------------------------------------------------------------------------- #
# Frame composition
# --------------------------------------------------------------------------- #
def _frame_title(frame: TransformedFrameData) -> str:
    parts = [
        f"{frame.dataset_name}",
        f"{frame.segment_id[:12]}",
        f"frame {frame.frame_id}",
        f"t={frame.timestamp:.2f}s",
    ]
    speed = _aux(frame).get("speed")
    if speed is not None:
        parts.append(f"speed={float(speed):.1f}")
    return "  |  ".join(parts)


def render_frame(fig: Figure, frame: TransformedFrameData) -> None:
    """Render ``frame`` onto ``fig`` (cleared first), auto-detecting modalities."""
    fig.clear()
    fig.patch.set_facecolor("black")
    cameras = _modality(frame, Modality.CAMERAS)
    has_cameras = isinstance(cameras, np.ndarray) or (
        isinstance(cameras, dict) and len(cameras) > 0
    )
    has_bev = any(_bev_grid(frame, m) for m in _RASTER_BEV_MODALITIES) or any(
        _modality(frame, m) is not None
        for m in (
            Modality.LIDAR_PC,
            Modality.DETECTIONS_3D,
            Modality.PAST_STATES,
            Modality.FUTURE_STATES,
            Modality.PREFERENCE_TRAJECTORY,
        )
    )

    if has_cameras and has_bev:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.04)
        _render_cameras(fig, gs[0, 0], cameras)
        _draw_bev(fig.add_subplot(gs[0, 1]), frame)
    elif has_cameras:
        _render_cameras(fig, fig.add_gridspec(1, 1)[0, 0], cameras)
    else:  # BEV only (or empty -> just ego/extent)
        _draw_bev(fig.add_subplot(fig.add_gridspec(1, 1)[0, 0]), frame)

    fig.suptitle(_frame_title(frame), fontsize=12, color="white")
    # ``subplots_adjust`` (not ``tight_layout``) -- the camera mosaic uses a
    # nested gridspec, which ``tight_layout`` warns about and mislays.
    fig.subplots_adjust(left=0.015, right=0.99, top=0.94, bottom=0.02, wspace=0.06)


def figure_to_bgr(fig: Figure) -> np.ndarray:
    """Rasterize ``fig`` to a contiguous ``(H, W, 3)`` uint8 BGR frame (for cv2)."""
    fig.canvas.draw()
    # buffer_rgba is provided by the Agg canvas (set at import); not in the
    # base-class type stub.
    rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
