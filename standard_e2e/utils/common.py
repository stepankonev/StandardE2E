import numpy as np
import yaml


def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def _assert_strictly_increasing(array: np.ndarray) -> None:
    """Validate that a 1D float array is strictly increasing.

    Accepts empty or single-element arrays. Raises ValueError otherwise when
    non-increasing (<=) adjacent differences are found.
    """
    if array.ndim != 1:
        raise ValueError("Array must be 1D.")
    if array.size <= 1:
        return
    if not np.all(np.diff(array) > 0):
        raise ValueError("Values must be strictly increasing.")


def _check_list_of_objects_or_none(objects: list | None, obj_type: type) -> None:
    if objects is None:
        return
    if not isinstance(objects, list):
        raise TypeError(
            f"objects must be a list of {obj_type} or None, got {type(objects)}"
        )
    if not all(isinstance(a, obj_type) for a in objects):
        raise TypeError(f"objects must be a list of {obj_type} or None")


def matrix_to_xyz_heading(
    pos_matrix: np.ndarray | list,
) -> tuple[float, float, float, float]:
    """
    Convert a 4x4 homogeneous transformation matrix into x, y, z, heading (yaw).
    Heading is rotation about the z-axis.
    """
    T = np.array(pos_matrix)
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    heading = np.arctan2(T[1, 0], T[0, 0])
    return x, y, z, heading
