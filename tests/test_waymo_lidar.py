"""Unit tests for the Waymo lidar range-image decoder.

Tests are pure-numpy; no Waymo proto fixture required. The four cases
gate the highest-bug-risk math in PR 2 step 2.2:

1. Byte-exact range-image decode (one beam, single valid range,
   identity extrinsic).
2. Identity extrinsic ego lift (multi-column, multiple valid ranges).
3. Non-trivial extrinsic translation lift.
4. Multi-laser concatenation produces one DataFrame with laser_id.
"""

from __future__ import annotations

import numpy as np
import pytest

from standard_e2e.data_structures import LidarData
from standard_e2e.utils.waymo_lidar import (
    LaserSpec,
    lasers_to_lidar_data,
    range_image_to_points_in_ego,
)


def _identity_extrinsic() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def test_range_image_byte_exact_one_beam_single_range():
    """One beam (H=1), W=4 columns, only column 0 has a valid range.

    Hand-computed: column 0 -> azimuth = 3*pi/4 (Waymo's "col 0 is rear")
    convention, beam inclination 0 -> horizontal, range=10. Expected:
        x = 10 * cos(0) * cos(3*pi/4) = -7.0710678
        y = 10 * cos(0) * sin(3*pi/4) =  7.0710678
        z = 0
    """
    range_image = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    inclinations = np.array([0.0], dtype=np.float32)
    pts = range_image_to_points_in_ego(
        range_image, _identity_extrinsic(), inclinations
    )
    assert pts.shape == (1, 3)
    assert pts.dtype == np.float32
    expected = np.array(
        [[-10.0 * np.sqrt(2) / 2, 10.0 * np.sqrt(2) / 2, 0.0]], dtype=np.float32
    )
    np.testing.assert_allclose(pts, expected, atol=1e-5)


def test_identity_extrinsic_preserves_polar_xyz():
    """With identity extrinsic, output XYZ equals the polar->cartesian
    formula directly (no rotation, no translation)."""
    H, W = 2, 8
    inclinations = np.array([0.0, 0.1], dtype=np.float32)
    rng = np.zeros((H, W), dtype=np.float32)
    # Sprinkle a few valid ranges
    rng[0, 0] = 5.0
    rng[1, 4] = 7.5
    rng[0, 6] = 12.0

    pts = range_image_to_points_in_ego(rng, _identity_extrinsic(), inclinations)
    # Hand-compute expected for each valid pixel using the same formula.
    col = np.arange(W)
    ratios = 1.0 - (col + 0.5) / W
    azimuth = (ratios * 2.0 - 1.0) * np.pi  # az_correction = 0 for identity
    expected_rows = []
    for r, c in [(0, 0), (0, 6), (1, 4)]:
        # iteration order matches np.where (row-major scan).
        rr = float(rng[r, c])
        inc = float(inclinations[r])
        az = float(azimuth[c])
        x = rr * np.cos(inc) * np.cos(az)
        y = rr * np.cos(inc) * np.sin(az)
        z = rr * np.sin(inc)
        expected_rows.append([x, y, z])
    expected = np.asarray(expected_rows, dtype=np.float32)
    assert pts.shape == expected.shape
    np.testing.assert_allclose(pts, expected, atol=1e-5)


def test_translation_only_extrinsic_shifts_points():
    """Extrinsic = pure translation (2, 0, 1.5). Output = polar XYZ + translation.

    Catches the 'forgot to apply extrinsic' bug.
    """
    range_image = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    inclinations = np.array([0.0], dtype=np.float32)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, 3] = [2.0, 0.0, 1.5]

    pts = range_image_to_points_in_ego(range_image, extrinsic, inclinations)
    expected = np.array(
        [
            [
                -10.0 * np.sqrt(2) / 2 + 2.0,
                10.0 * np.sqrt(2) / 2 + 0.0,
                0.0 + 1.5,
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pts, expected, atol=1e-5)


def test_five_laser_concatenation_preserves_laser_id():
    """5 lasers with disjoint single-point range images concatenate into
    one DataFrame with all 5 distinct laser_id values."""
    extrinsic = _identity_extrinsic()
    inclinations = np.array([0.0], dtype=np.float32)

    laser_ids = [1, 2, 3, 4, 5]  # TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR
    specs = []
    for i, lid in enumerate(laser_ids):
        # Each laser has a single valid point at column i.
        rng = np.zeros((1, 5), dtype=np.float32)
        rng[0, i] = 10.0 + i
        specs.append(
            LaserSpec(
                laser_id=lid,
                range_image=rng,
                extrinsic=extrinsic,
                beam_inclinations=inclinations,
            )
        )

    result = lasers_to_lidar_data(specs)
    assert isinstance(result, LidarData)
    df = result.points
    assert len(df) == 5
    assert set(df["laser_id"].unique().tolist()) == set(laser_ids)
    for col in ("x", "y", "z", "laser_id"):
        assert col in df.columns


def test_range_image_validates_input_shapes():
    extrinsic = _identity_extrinsic()
    inclinations = np.array([0.0], dtype=np.float32)
    # Wrong inclination length (H=1, but provide 2 inclinations)
    rng = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        range_image_to_points_in_ego(rng, extrinsic, np.array([0.0, 0.1], dtype=np.float32))
    # Wrong extrinsic shape
    with pytest.raises(ValueError):
        range_image_to_points_in_ego(rng, np.eye(3, dtype=np.float32), inclinations)
