import numpy as np
import pytest

from standard_e2e.utils.geometry import (
    intrinsics_matrix,
    quat_wxyz_to_rotmat,
    quats_wxyz_to_rotmats,
    se3,
    transform_points,
    wrap_to_pi,
)


def _manual_wxyz(q):
    """The hand-rolled Hamilton (w, x, y, z) formula the datasets used to carry."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_quats_match_hand_rolled_hamilton_formula():
    # Regression guard: the scipy-backed helper must reproduce the manual
    # formula comma2k19 / wayve_scenes previously hand-rolled (whose convention
    # is empirically locked), to bit-level tolerance.
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rotmats = quats_wxyz_to_rotmats(q)
    assert rotmats.shape == (64, 3, 3)
    for i in range(len(q)):
        np.testing.assert_allclose(rotmats[i], _manual_wxyz(q[i]), atol=1e-12)


def test_quat_wxyz_identity_and_known_rotation():
    np.testing.assert_allclose(quat_wxyz_to_rotmat([1, 0, 0, 0]), np.eye(3), atol=1e-12)
    s = np.sqrt(0.5)  # +90 deg about z: (w,x,y,z) = (cos45, 0, 0, sin45)
    np.testing.assert_allclose(
        quat_wxyz_to_rotmat([s, 0, 0, s]),
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )


def test_quats_shape_and_norm_validation():
    with pytest.raises(ValueError):
        quats_wxyz_to_rotmats(np.zeros((1, 3)))  # wrong shape
    with pytest.raises(ValueError):
        quats_wxyz_to_rotmats(np.zeros((1, 4)))  # zero-norm (scipy rejects)


def test_se3_assembles_and_respects_dtype():
    r = quat_wxyz_to_rotmat([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    t = [1.0, 2.0, 3.0]
    transform = se3(r, t)
    assert transform.shape == (4, 4)
    np.testing.assert_allclose(transform[:3, :3], r)
    np.testing.assert_allclose(transform[:3, 3], t)
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
    assert se3(np.eye(3), [0, 0, 0], dtype=np.float32).dtype == np.float32


def test_intrinsics_matrix():
    k = intrinsics_matrix(910, 910, 582, 437)
    np.testing.assert_allclose(k, [[910, 0, 582], [0, 910, 437], [0, 0, 1]])
    assert k.dtype == np.float32


def test_transform_points():
    # +90 deg about z, then translate by +x.
    transform = se3([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [1, 0, 0])
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    out = transform_points(transform, pts)
    np.testing.assert_allclose(out, [[1, 1, 0], [0, 0, 0]], atol=1e-6)
    assert out.dtype == np.float32  # computed in the points' dtype
    with pytest.raises(ValueError):
        transform_points(transform, np.zeros((4, 2)))


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == 0.0
    np.testing.assert_allclose(wrap_to_pi(np.pi / 2), np.pi / 2, atol=1e-12)
    # Angles outside [-pi, pi] wrap back into range.
    np.testing.assert_allclose(wrap_to_pi(3 * np.pi / 2), -np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(wrap_to_pi(-3 * np.pi / 2), np.pi / 2, atol=1e-12)
    assert abs(wrap_to_pi(5 * np.pi)) == pytest.approx(np.pi)
    assert isinstance(wrap_to_pi(1.0), float)
