"""Source-dataset converter for WayveScenes101.

Iterates extracted scenes frame-by-frame. Expected on-disk layout::

    <input_path>/scene_<NNN>/
        colmap_sparse/rig/{cameras.bin, images.bin, points3D.bin}
        images/<camera>/<timestamp_ns>.jpeg
        masks/<camera>/<timestamp_ns>.png

One frame = one timestamp (that has the reference camera); one scene = one
segment. Yielding ``(scene_dir, timestamp_ns)`` ordered by scene then
timestamp keeps the processor's per-scene COLMAP cache warm across each
multiprocessing chunk.

``split`` is a passthrough output label (WayveScenes101 has no perception
split). Scenes are read directly from ``input_path``; if an ``input_path/split``
directory exists it is preferred. Scenes must already be extracted from the
distributed ``scene_<NNN>.zip`` archives.

With ``STANDARD_E2E_DEBUG=true`` only the first scene is processed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

from standard_e2e.caching import SourceDatasetConverter
from standard_e2e.caching.src_datasets.wayve_scenes._colmap import read_images_bin


class WayveScenesDatasetConverter(SourceDatasetConverter):
    """Iterates WayveScenes101 scenes frame-by-frame."""

    @property
    def multiprocessing_start_method(self) -> str:
        # Worker hot path is cv2 (JPEG decode) + numpy only; no TF ops fire in
        # the worker, so ``fork`` is safe and avoids the spawn import tax.
        return "fork"

    def _scene_root(self) -> Path:
        split_root = Path(self._input_path) / self._split
        return split_root if split_root.is_dir() else Path(self._input_path)

    def _get_source_dataset_iterator(self) -> Iterator[tuple[Path, int]]:
        root = self._scene_root()
        if not root.is_dir():
            raise FileNotFoundError(f"WayveScenes input dir not found: {root}")
        scene_dirs = sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "colmap_sparse" / "rig" / "images.bin").is_file()
        )
        if not scene_dirs:
            raise FileNotFoundError(
                f"No extracted WayveScenes scenes under {root} "
                "(expected scene_*/colmap_sparse/rig/images.bin)"
            )
        if os.environ.get("STANDARD_E2E_DEBUG", "false").lower() == "true":
            logging.info("STANDARD_E2E_DEBUG: processing only the first scene.")
            scene_dirs = scene_dirs[:1]
        logging.info("Found %d WayveScenes scene(s).", len(scene_dirs))

        # Pre-list (scene, timestamp) tuples up front. The per-scene images.bin
        # read here is the cheap (pose-only) parse; it consolidates the scene
        # scan into one pass rather than interleaving with worker reads.
        items: list[tuple[Path, int]] = []
        for scene_dir in scene_dirs:
            rig = scene_dir / "colmap_sparse" / "rig"
            timestamps = sorted(
                {
                    int(Path(img.name).stem)
                    for img in read_images_bin(rig / "images.bin")
                    if Path(img.name).parent.name == "front-forward"
                }
            )
            items.extend((scene_dir, ts) for ts in timestamps)
        logging.info("Pre-listed %d WayveScenes frames.", len(items))
        return iter(items)
