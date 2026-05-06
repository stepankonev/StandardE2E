"""Tiny binary-PCD reader scoped to NAVSIM's MergedPointCloud layout.

NAVSIM ships merged lidar sweeps as binary `.pcd` v0.7 files whose header
declares ``FIELDS x y z intensity lidar_info ring`` with sizes ``4 4 4 1
1 1`` and types ``F F F U U U``. We only need the XYZ coordinates for the
unified format, but we keep the parser slightly general (parses the
header, supports extra fields by skipping them) so future datasets that
ship PCDs with the same layout can reuse it.

We do NOT depend on ``pypcd`` / ``open3d`` / ``nuplan-devkit``: those
either pull heavy transitive deps or are unmaintained on Python 3.12.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

_PCD_DTYPE_LOOKUP: Final[dict[tuple[str, int], str]] = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
}


def read_navsim_pcd_xyz(path: Path) -> np.ndarray:
    """Return ``(N, 3)`` float32 XYZ from a NAVSIM binary PCD file.

    The header is ASCII, terminated by ``DATA binary\\n``; everything after
    that is a packed numpy structured array with the per-field dtype declared
    in the header. We parse the header, build a numpy ``dtype`` from
    ``FIELDS``/``SIZE``/``TYPE``/``COUNT``, then slice ``[x, y, z]``.
    """
    with open(path, "rb") as fp:
        raw = fp.read()
    header_end = raw.find(b"DATA binary\n")
    if header_end == -1:
        raise ValueError(f"Not a binary PCD: {path}")
    header_bytes = raw[:header_end]
    payload = raw[header_end + len(b"DATA binary\n") :]

    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    n_points = 0
    for line in header_bytes.decode("ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, *vals = line.split()
        if key == "FIELDS":
            fields = vals
        elif key == "SIZE":
            sizes = [int(v) for v in vals]
        elif key == "TYPE":
            types = vals
        elif key == "COUNT":
            counts = [int(v) for v in vals]
        elif key == "POINTS":
            n_points = int(vals[0])

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(
            f"Inconsistent PCD header in {path}: "
            f"FIELDS({len(fields)})/SIZE({len(sizes)})/TYPE({len(types)})/"
            f"COUNT({len(counts)})"
        )

    dtype_pairs: list[tuple[str, str]] = []
    for name, sz, tp, ct in zip(fields, sizes, types, counts):
        if ct != 1:
            raise ValueError(
                f"PCD field {name!r} has count={ct}; only count=1 is supported."
            )
        np_dtype = _PCD_DTYPE_LOOKUP.get((tp, sz))
        if np_dtype is None:
            raise ValueError(f"Unsupported PCD dtype: type={tp} size={sz}")
        dtype_pairs.append((name, np_dtype))

    arr = np.frombuffer(payload, dtype=np.dtype(dtype_pairs), count=n_points)
    out = np.empty((n_points, 3), dtype=np.float32)
    out[:, 0] = arr["x"]
    out[:, 1] = arr["y"]
    out[:, 2] = arr["z"]
    return out
