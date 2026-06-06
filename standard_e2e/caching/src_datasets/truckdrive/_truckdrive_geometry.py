"""Calibration / pose geometry helpers for TruckDrive.

TruckDrive ships extrinsics as a ROS-style **static transform tree**
(``calibrations/calib_tf_tree_full.json``): a flat dict whose entries each
carry ``header.frame_id`` (parent), ``child_frame_id`` (child) and a
``transform`` (translation + ``(x, y, z, w)`` quaternion). The tree is rooted
at ``vehicle`` and chains ``vehicle -> cab -> <sensor>`` (the truck is
articulated, so cab-mounted sensors hang off the ``cab`` frame). To express any
sensor's points in another frame we build an undirected graph of the static
transforms and BFS for a path between two frames.

The frame convention matches the official devkit (``vis_utils.load_utils``):
:func:`find_transform` returns ``T_tgt_from_src`` -- the 4x4 that maps a *point*
expressed in ``src`` coordinates to its coordinates in ``tgt`` (so projecting a
sensor's points into a camera is ``cam2img @ find_transform(graph, sensor,
camera)``). We reproduce the devkit's edge math exactly so calibrations resolve
identically.

Ego frame: StandardE2E expresses per-frame sensors / boxes in the dataset's
canonical annotation frame, ``velodyne`` (the frame the public 3D boxes and the
devkit viewer use; FLU -- x-forward, y-left, z-up). Lidar is moved into it from
the Aeva reference frame; camera extrinsics are ``T_velodyne_from_camera``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from standard_e2e.utils import quat_wxyz_to_rotmat, se3

# TF-tree node names used as anchors (see ``calib_tf_tree_full.json`` and the
# devkit ``dataset_details.py``).
VELODYNE_FRAME = "velodyne"
AEVA_REFERENCE_FRAME = "lidar_aeva_forward_center_wide"


def _transform_to_matrix(transform: dict[str, Any]) -> np.ndarray:
    """Convert one TF-tree ``transform`` block to a 4x4 (parent_from_child).

    ``transform`` has ``translation`` ``{x, y, z}`` and a unit quaternion
    ``rotation`` ``{x, y, z, w}`` (the child frame's pose in the parent frame).
    """
    t = transform["translation"]
    r = transform["rotation"]
    rotation = quat_wxyz_to_rotmat((r["w"], r["x"], r["y"], r["z"]))
    return se3(rotation, (t["x"], t["y"], t["z"]), dtype=np.float64)


def build_tf_graph(tf_tree: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Build an undirected graph of static transforms from a TF tree.

    ``graph[a][b]`` is the 4x4 that maps a point in frame ``a`` to frame ``b``.
    Mirrors the devkit's ``build_graph_of_transforms`` edge assignment so paths
    resolve to identical matrices.
    """
    graph: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for entry in tf_tree.values():
        parent = entry["header"]["frame_id"]
        child = entry["child_frame_id"]
        parent_from_child = _transform_to_matrix(entry["transform"])
        # Edge child->parent maps a child-frame point into the parent frame.
        graph[child][parent] = parent_from_child
        graph[parent][child] = np.linalg.inv(parent_from_child)
    return graph


def find_transform(
    graph: dict[str, dict[str, np.ndarray]], src: str, tgt: str
) -> np.ndarray:
    """Return ``T_tgt_from_src``: the 4x4 mapping a point in ``src`` to ``tgt``.

    BFS over the static-transform graph (the tree is small, a dozen-ish nodes).
    Raises ``KeyError`` if either frame is absent and ``ValueError`` if the
    frames are disconnected.
    """
    if src not in graph:
        raise KeyError(f"source frame {src!r} not in transform tree")
    if tgt not in graph:
        raise KeyError(f"target frame {tgt!r} not in transform tree")
    visited: set[str] = set()
    queue: deque[tuple[str, np.ndarray]] = deque([(src, np.eye(4, dtype=np.float64))])
    while queue:
        current, tgt_from_current_src = queue.popleft()
        if current == tgt:
            return tgt_from_current_src
        visited.add(current)
        for neighbour, neighbour_from_current in graph[current].items():
            if neighbour not in visited:
                queue.append((neighbour, neighbour_from_current @ tgt_from_current_src))
    raise ValueError(f"no path from {src!r} to {tgt!r} in transform tree")


def pose_world_from_ego(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Assemble ``T_world_from_ego`` from a ``gt_trajectory`` row.

    ``gt_trajectory.txt`` stores the ego pose as a translation ``(X, Y, Z)`` and
    a scalar-last quaternion ``(R_X, R_Y, R_Z, R_W)`` in a per-scene local world
    frame (anchored at the first frame). Returns a ``(4, 4)`` float64 transform
    mapping ego-frame points to that world frame.
    """
    qx, qy, qz, qw = (float(v) for v in np.asarray(quat_xyzw, dtype=np.float64))
    rotation = quat_wxyz_to_rotmat((qw, qx, qy, qz))
    return se3(rotation, np.asarray(position_xyz, dtype=np.float64), dtype=np.float64)
