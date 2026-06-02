"""Tests for the dependency-free COLMAP binary reader used by WayveScenes.

Two layers:
* **Synthetic round-trip** (always runs): write tiny ``cameras.bin`` /
  ``images.bin`` / ``points3D.bin`` with known values — including 2D-point and
  track blobs that the reader must *skip* — and assert it reads back exactly.
* **Parity vs pycolmap** (skipped unless pycolmap + the real dataset are
  present): the ground-truth check that the reader matches the reference
  implementation on a real scene.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from standard_e2e.caching.src_datasets.wayve_scenes._colmap import (
    qvec_wxyz_to_rotmat,
    read_cameras_bin,
    read_images_bin,
    read_points3D_bin,
)

# Extracted WayveScenes scenes: ``WAYVE_SCENES_ROOT`` env var wins; the raw
# mount is a fallback (it normally holds only .zip archives, so tests skip in
# CI / wherever scenes are not extracted).
_WAYVE_ROOTS = [
    Path(p)
    for p in (
        os.environ.get("WAYVE_SCENES_ROOT", ""),
        "/mnt/bigdisk/datasets/wayve_scenes_unzipped",
    )
    if p
]


# --------------------------------------------------------------------------- #
# Synthetic binary writers (match the reader's format exactly)
# --------------------------------------------------------------------------- #
def _write_cameras(path: Path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, model, w, h, params in cams:
            f.write(struct.pack("<iiQQ", cid, model, w, h))
            f.write(struct.pack("<" + "d" * len(params), *params))


def _write_images(path: Path, imgs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for image_id, qvec, tvec, camera_id, name, n_pts2d in imgs:
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<ddddddd", *qvec, *tvec))
            f.write(struct.pack("<i", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", n_pts2d))
            # 2D points the reader must skip: (x f64, y f64, id u64) each
            for k in range(n_pts2d):
                f.write(struct.pack("<ddQ", float(k), float(-k), k))


def _write_points3d(path: Path, pts):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz, rgb, err, track in pts:
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *rgb))
            f.write(struct.pack("<d", err))
            f.write(struct.pack("<Q", len(track)))
            for image_id, p2d_idx in track:  # the track the reader must skip
                f.write(struct.pack("<II", image_id, p2d_idx))


# --------------------------------------------------------------------------- #
# Synthetic round-trip tests
# --------------------------------------------------------------------------- #
def test_read_cameras_bin_roundtrip(tmp_path):
    cams = [
        (1, 5, 1920, 1080, [900.0, 905.0, 960.0, 540.0, 0.1, -0.2, 0.3, -0.4]),
        (2, 1, 100, 200, [1.0, 2.0, 3.0, 4.0]),  # PINHOLE, 4 params
    ]
    p = tmp_path / "cameras.bin"
    _write_cameras(p, cams)
    out = read_cameras_bin(p)
    assert set(out) == {1, 2}
    assert out[1].model_id == 5 and out[1].width == 1920 and out[1].height == 1080
    np.testing.assert_allclose(out[1].params, cams[0][4])
    np.testing.assert_allclose(out[2].params, cams[1][4])


def test_read_images_bin_skips_2d_points(tmp_path):
    # Two images with DIFFERENT numbers of 2D points: the reader must skip the
    # first image's blob to land correctly on the second image's record.
    imgs = [
        (10, [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1, "front-forward/111.jpeg", 7),
        (11, [0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 6.0], 2, "left-forward/222.jpeg", 3),
    ]
    p = tmp_path / "images.bin"
    _write_images(p, imgs)
    out = read_images_bin(p)
    assert [im.image_id for im in out] == [10, 11]
    assert out[0].name == "front-forward/111.jpeg"
    assert out[1].name == "left-forward/222.jpeg"
    np.testing.assert_allclose(out[0].qvec_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1].tvec, [4.0, 5.0, 6.0])
    assert out[1].camera_id == 2


def test_read_points3d_bin_skips_tracks(tmp_path):
    pts = [
        (100, (1.0, 2.0, 3.0), (10, 20, 30), 0.5, [(1, 0), (2, 1)]),
        (101, (-4.0, -5.0, -6.0), (40, 50, 60), 9.9, [(3, 2)]),
    ]
    p = tmp_path / "points3D.bin"
    _write_points3d(p, pts)
    xyz, rgb, error, track_len = read_points3D_bin(p)
    assert xyz.shape == (2, 3) and rgb.shape == (2, 3)
    np.testing.assert_allclose(xyz[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(xyz[1], [-4.0, -5.0, -6.0])
    np.testing.assert_array_equal(rgb[1], [40, 50, 60])
    np.testing.assert_allclose(error, [0.5, 9.9])
    np.testing.assert_array_equal(track_len, [2, 1])


def test_qvec_to_rotmat_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_xyzw = rng.normal(size=4)
        q_xyzw /= np.linalg.norm(q_xyzw)
        from scipy.spatial.transform import Rotation

        ref = Rotation.from_quat(q_xyzw).as_matrix()
        ours = qvec_wxyz_to_rotmat(np.array([q_xyzw[3], *q_xyzw[:3]]))
        np.testing.assert_allclose(ours, ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# Parity vs pycolmap on the real dataset (skipped when either is missing)
# --------------------------------------------------------------------------- #
def _find_real_rig() -> Path | None:
    for root in _WAYVE_ROOTS:
        if not root.is_dir():
            continue
        for scene in sorted(root.glob("scene_*")):
            rig = scene / "colmap_sparse" / "rig"
            if (rig / "images.bin").is_file():
                return rig
    return None


def test_parity_with_pycolmap_on_real_scene():
    pycolmap = pytest.importorskip("pycolmap")
    rig = _find_real_rig()
    if rig is None:
        pytest.skip("no extracted WayveScenes scene available")
    from scipy.spatial.transform import Rotation

    imgs = read_images_bin(rig / "images.bin")
    cams = read_cameras_bin(rig / "cameras.bin")
    xyz, _, _, _ = read_points3D_bin(rig / "points3D.bin")

    rec = pycolmap.Reconstruction(str(rig))
    assert len(cams) == len(rec.cameras)
    assert len(imgs) == len(rec.images)
    assert len(xyz) == len(rec.points3D)

    our_by_name = {im.name: im for im in imgs}
    mismatches = 0
    for ref in rec.images.values():
        cfw = ref.cam_from_world()
        ref_R = Rotation.from_quat(np.asarray(cfw.rotation.quat)).as_matrix()
        o = our_by_name[ref.name]
        if not np.allclose(qvec_wxyz_to_rotmat(o.qvec_wxyz), ref_R, atol=1e-9):
            mismatches += 1
        if not np.allclose(o.tvec, np.asarray(cfw.translation), atol=1e-9):
            mismatches += 1
    assert mismatches == 0
