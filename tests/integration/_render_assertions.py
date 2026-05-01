"""Content-based assertions for visual-gate PNG renders.

The PR-gate visual smokes used to bound output size with a 5 KB - 2 MB
band. That band catches truly broken outputs (zero bytes, multi-MB
blobs) but passes a regressed-to-blank renderer that emits a 50 KB
solid-gray PNG. This module replaces the size-band check with a
content-based check that distinguishes a real render from a blank one
without requiring a human in the loop. Visual verification still
ultimately means humans-look-at-PNGs; this is just the *automated* floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# A constant-color blank render has 1 distinct color. The most
# palette-restricted real render currently produced by ``render_frame``
# is the intent vis (256 colors). 100 is therefore well above any
# blank/near-blank output and well below any real one.
_MIN_DISTINCT_COLORS = 100

# Per-channel variance floor: a constant image has variance 0; the
# intent vis (mostly gray + a small arrow + small text) has variance
# ~1300 per channel; richer renders sit at 3000+. 500 rejects regressed-
# to-blank renders while preserving headroom for minimal-but-real ones.
_MIN_PER_CHANNEL_VARIANCE = 500.0


def assert_png_has_real_content(path: Union[str, Path]) -> None:
    """Assert the PNG at ``path`` has non-trivial visual content.

    Two complementary checks:

    * ``distinct_colors >= 100`` rejects pure constant outputs and
      almost-constant ones (e.g. a uniform gray with a single text
      string overlay).
    * ``min(per_channel_variance) > 500`` rejects renders that have
      many distinct colors but trivially low spatial variation
      (e.g. a smooth low-amplitude gradient with no real content).

    Either check alone leaves a hole the other closes; running both
    drops the surviving failure modes to negligibly thin slices.

    Raises:
        AssertionError: if either content threshold is violated.
    """
    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img)
    flat = arr.reshape(-1, arr.shape[-1])
    # Distinct colors: pack each (R, G, B) into a structured-row view
    # so np.unique counts unique pixels rather than unique scalars.
    distinct = int(
        len(np.unique(flat.view([("", flat.dtype)] * flat.shape[1])))
    )
    assert distinct >= _MIN_DISTINCT_COLORS, (
        f"{path}: only {distinct} distinct colors "
        f"(<{_MIN_DISTINCT_COLORS}); PNG appears blank or near-blank."
    )
    var_per_channel = arr.astype(np.float64).var(axis=(0, 1))
    min_var = float(var_per_channel.min())
    assert min_var > _MIN_PER_CHANNEL_VARIANCE, (
        f"{path}: minimum per-channel variance {min_var:.1f} "
        f"<= {_MIN_PER_CHANNEL_VARIANCE}; PNG appears flat/featureless."
    )
