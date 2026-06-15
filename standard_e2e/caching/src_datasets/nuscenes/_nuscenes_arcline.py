"""Lane-centerline discretization for nuScenes maps, vendored from the devkit.

nuScenes stores each lane / lane-connector centerline as an ``arcline_path_3``
record -- a sequence of arc/line segments (start pose, shape, radius, segment
lengths) -- rather than an explicit polyline. Recovering the centerline polyline
requires integrating those segments, which the devkit does in
``nuscenes/map_expansion/arcline_path_utils.py``. We can't depend on the devkit
at runtime (it pins ``numpy<2`` / ``Shapely~=2.0.3`` against this project's numpy
2.x), and these functions are pure-Python ``math`` (no numpy), so the few we need
(``discretize_lane`` and its helpers) are vendored here verbatim.

Source: nuscenes-devkit ``python-sdk/nuscenes/map_expansion/arcline_path_utils.py``
(Freddy Boulton, 2020), licensed under the Apache License 2.0, Copyright nuScenes
devkit authors (Motional). Only ``discretize_lane`` and its dependencies are
copied; the numpy-dependent projection/curvature helpers are omitted.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

# (x, y, yaw) in the map (global) frame.
Pose = Tuple[float, float, float]
ArcLinePath = Dict[str, Any]


def principal_value(angle_in_radians: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    interval_min = -math.pi
    two_pi = 2 * math.pi
    return (angle_in_radians - interval_min) % two_pi + interval_min


def compute_segment_sign(arcline_path: ArcLinePath) -> Tuple[int, int, int]:
    """Per-segment turn sign from the path shape (0 straight, -1 right, 1 left)."""
    shape = arcline_path["shape"]
    segment_sign = [0, 0, 0]

    if shape in ("LRL", "LSL", "LSR"):
        segment_sign[0] = 1
    else:
        segment_sign[0] = -1

    if shape == "RLR":
        segment_sign[1] = 1
    elif shape == "LRL":
        segment_sign[1] = -1
    else:
        segment_sign[1] = 0

    if shape in ("LRL", "LSL", "RSL"):
        segment_sign[2] = 1
    else:
        segment_sign[2] = -1

    return segment_sign[0], segment_sign[1], segment_sign[2]


def get_transformation_at_step(pose: Pose, step: float) -> Pose:
    """Affine transformation ``step`` meters along a constant-curvature segment."""
    theta = pose[2] * step
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    if abs(pose[2]) < 1e-6:
        return pose[0] * step, pose[1] * step, theta

    new_x = (pose[1] * (ctheta - 1.0) + pose[0] * stheta) / pose[2]
    new_y = (pose[0] * (1.0 - ctheta) + pose[1] * stheta) / pose[2]
    return new_x, new_y, theta


def apply_affine_transformation(pose: Pose, transformation: Pose) -> Pose:
    """Apply ``transformation`` (a pose delta) to a starting ``pose``."""
    new_x = (
        math.cos(pose[2]) * transformation[0]
        - math.sin(pose[2]) * transformation[1]
        + pose[0]
    )
    new_y = (
        math.sin(pose[2]) * transformation[0]
        + math.cos(pose[2]) * transformation[1]
        + pose[1]
    )
    new_yaw = principal_value(pose[2] + transformation[2])
    return new_x, new_y, new_yaw


def _get_lie_algebra(
    segment_sign: Tuple[int, int, int], radius: float
) -> List[Tuple[float, float, float]]:
    """Lie-algebra generators for the three segments of an arcline path."""
    return [
        (1.0, 0.0, segment_sign[0] / radius),
        (1.0, 0.0, segment_sign[1] / radius),
        (1.0, 0.0, segment_sign[2] / radius),
    ]


def pose_at_length(arcline_path: ArcLinePath, pos: float) -> Pose:
    """Pose ``pos`` meters along a single arcline path."""
    path_length = sum(arcline_path["segment_length"])
    assert 1e-6 <= pos
    pos = max(0.0, min(pos, path_length))

    result: Pose = arcline_path["start_pose"]
    segment_sign = compute_segment_sign(arcline_path)
    break_points = _get_lie_algebra(segment_sign, arcline_path["radius"])

    for i in range(len(break_points)):
        length = arcline_path["segment_length"][i]
        if pos <= length:
            transformation = get_transformation_at_step(break_points[i], pos)
            result = apply_affine_transformation(result, transformation)
            break
        transformation = get_transformation_at_step(break_points[i], length)
        result = apply_affine_transformation(result, transformation)
        pos -= length

    return result


def discretize(arcline_path: ArcLinePath, resolution_meters: float) -> List[Pose]:
    """Discretize a single arcline path into a list of ``(x, y, yaw)`` poses."""
    path_length = sum(arcline_path["segment_length"])
    radius = arcline_path["radius"]

    n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))
    resolution_meters = path_length / (n_points - 1)

    discretization: List[Pose] = []
    cumulative_length = [
        arcline_path["segment_length"][0],
        arcline_path["segment_length"][0] + arcline_path["segment_length"][1],
        path_length + resolution_meters,
    ]
    segment_sign = compute_segment_sign(arcline_path)
    poses = _get_lie_algebra(segment_sign, radius)
    temp_pose = arcline_path["start_pose"]

    g_i = 0
    g_s = 0.0
    for step in range(n_points):
        step_along_path = step * resolution_meters
        if step_along_path > cumulative_length[g_i]:
            temp_pose = pose_at_length(arcline_path, step_along_path)
            g_s = step_along_path
            g_i += 1
        transformation = get_transformation_at_step(poses[g_i], step_along_path - g_s)
        new_pose = apply_affine_transformation(temp_pose, transformation)
        discretization.append(new_pose)

    return discretization


def discretize_lane(lane: List[ArcLinePath], resolution_meters: float) -> List[Pose]:
    """Discretize a lane (list of arcline paths) into ``(x, y, yaw)`` poses."""
    pose_list: List[Pose] = []
    for path in lane:
        pose_list.extend(discretize(path, resolution_meters))
    return pose_list
