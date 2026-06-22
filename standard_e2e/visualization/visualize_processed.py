"""CLI: render videos from a processed dataset folder.

Consumes a folder of processed output (the ``index.parquet`` + ``.npz`` frames a
:class:`~standard_e2e.caching.source_dataset_converter.SourceDatasetConverter`
writes, plus the ``dataset_info.yaml`` descriptor) and produces one MP4 per
scene. Modalities are auto-detected per frame -- the tool is dataset-agnostic.

Scene selection (mutually exclusive; default = first scene):
    --scene-id ID [--scene-id ID ...]   render exactly these segment(s)
    --num-scenes N                       render the first N segments (index order)

Example::

    python -m standard_e2e.visualization.visualize_processed \\
        /data/out/kitscenes_multimodal/val --num-scenes 2 --out /tmp/viz
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from standard_e2e.constants import DATASET_INFO_FILE_NAME, INDEX_FILE_NAME
from standard_e2e.data_structures.frame_data import TransformedFrameData
from standard_e2e.visualization.render import figure_to_bgr, render_frame

_MAX_INDEX_SEARCH_DEPTH = 3


def _find_index(input_path: str) -> str:
    """Locate ``index.parquet`` at or (a few levels) below ``input_path``."""
    if os.path.isfile(input_path) and input_path.endswith(".parquet"):
        return input_path
    direct = os.path.join(input_path, INDEX_FILE_NAME)
    if os.path.isfile(direct):
        return direct
    pattern = input_path
    for _ in range(_MAX_INDEX_SEARCH_DEPTH):
        pattern = os.path.join(pattern, "*")
        import glob

        hits = sorted(glob.glob(os.path.join(pattern, INDEX_FILE_NAME)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No {INDEX_FILE_NAME} found at or below {input_path}. Point --input_path "
        f"at a processed <dataset>/<split>/ folder."
    )


def _load_dataset_info(index_dir: str) -> Optional[dict]:
    path = os.path.join(index_dir, DATASET_INFO_FILE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else None


def _select_segments(
    index: pd.DataFrame, scene_ids: Optional[list[str]], num_scenes: Optional[int]
) -> list[str]:
    """Resolve the requested segment ids (default: the first segment)."""
    available = list(dict.fromkeys(index["segment_id"].tolist()))  # index order
    if scene_ids:
        missing = [s for s in scene_ids if s not in set(available)]
        if missing:
            raise ValueError(
                f"scene id(s) not in index: {missing}. Available (first few): "
                f"{available[:5]}{' ...' if len(available) > 5 else ''}"
            )
        return scene_ids
    n = num_scenes if num_scenes is not None else 1
    return available[:n]


def _render_segment(
    fig,
    index_dir: str,
    segment_rows: pd.DataFrame,
    out_path: str,
    fps: int,
    max_frames: Optional[int],
) -> int:
    """Render one segment's frames (frame-ascending) to ``out_path``. Returns the
    number of frames written."""
    rows = segment_rows.sort_values("frame_id")
    if max_frames:
        rows = rows.head(max_frames)
    writer = None
    written = 0
    for _, row in rows.iterrows():
        npz_path = os.path.join(index_dir, os.path.basename(row["filename"]))
        if not os.path.isfile(npz_path):
            logging.warning("missing npz, skipping: %s", npz_path)
            continue
        frame = TransformedFrameData.from_npz(npz_path)
        render_frame(fig, frame)
        bgr = figure_to_bgr(fig)
        if writer is None:
            height, width = bgr.shape[:2]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        writer.write(bgr)
        written += 1
    if writer is not None:
        writer.release()
    return written


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Processed <dataset>/<split>/ folder (containing index.parquet).",
    )
    parser.add_argument("--out", default="visualizations", help="Output directory.")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Cap frames rendered per scene."
    )
    parser.add_argument(
        "--cam-width", type=int, default=22, help="Figure width in inches."
    )
    parser.add_argument(
        "--fig-height", type=int, default=9, help="Figure height in inches."
    )
    select = parser.add_mutually_exclusive_group()
    select.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Segment id to render (repeatable). Mutually exclusive with --num-scenes.",
    )
    select.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Render the first N scenes (index order). Default: 1.",
    )
    args = parser.parse_args(argv)

    index_path = _find_index(args.input_path)
    index_dir = os.path.dirname(index_path)
    index = pd.read_parquet(index_path)
    if index.empty:
        raise ValueError(f"empty index: {index_path}")

    info = _load_dataset_info(index_dir)
    if info:
        logging.info(
            "dataset=%s split=%s adapters=%s",
            info.get("dataset_name"),
            info.get("split"),
            [a.get("name") for a in info.get("adapters", [])],
        )

    segments = _select_segments(index, args.scene_id, args.num_scenes)
    logging.info("Rendering %d scene(s): %s", len(segments), segments)

    fig = plt.figure(figsize=(args.cam_width, args.fig_height))
    total = 0
    written_paths: list[str] = []
    for segment_id in segments:
        rows = index[index["segment_id"] == segment_id]
        dataset_name = str(rows.iloc[0]["dataset_name"])
        split = str(rows.iloc[0]["split"])
        safe = str(segment_id).replace("/", "_")
        out_path = os.path.join(args.out, f"{dataset_name}_{split}_{safe}.mp4")
        n = _render_segment(fig, index_dir, rows, out_path, args.fps, args.max_frames)
        if n:
            written_paths.append(out_path)
            total += n
            logging.info("wrote %s (%d frames)", out_path, n)
    plt.close(fig)
    logging.info("Done: %d frame(s) across %d video(s).", total, len(written_paths))
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
