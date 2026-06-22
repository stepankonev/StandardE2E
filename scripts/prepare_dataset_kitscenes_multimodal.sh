#!/usr/bin/env bash
# KITScenes *Multimodal* (the dataset key is `kitscenes_multimodal`, distinct from
# the planned KITScenes-LongTail variant) must be DOWNLOADED and EXTRACTED first.
#
# 1. Download a split from Hugging Face (gated; accept the terms first), e.g.:
#      hf download KIT-MRT/KITScenes-Multimodal --repo-type dataset \
#          --include "data/val/*" --local-dir "$ARCHIVE_SRC"
#    (split folders on HF are train/val/test/test-e2e/overlap_train_val; or use
#    the KIT-MRT/KITScenes-Multimodal-Sample repo for a single scene.)
# 2. Extract the per-scene tarballs (scripts/extract_kitscenes.sh) so each
#    <scene_uuid>/ directory exists.
# 3. --input_path points at the directory holding the scenes -- the processor
#    reads <scene_uuid>/{calibration,poses.txt,timestamp.reference.txt,
#    camera_ring_*,lidar_top,maps}.

ARCHIVE_SRC="/mnt/bigdisk/kitscenes_multimodal_dl"          # dir holding downloaded tars
SCENES_PATH="/mnt/nvme1/data/kitscenes_multimodal_scenes"   # extracted scenes root
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_kitscenes_multimodal"
# train / val / test / test_e2e / overlap_train_val / all. With a flat layout
# (e.g. the single-scene sample) every scene is processed and SPLIT is just the
# output label.
SPLIT="val"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/kitscenes_multimodal.yaml"

# One-time extraction (idempotent). Skip if you already have a flat scenes dir
# (such as the downloaded sample) -- just set SCENES_PATH to it.
bash "$SCRIPT_DIR/extract_kitscenes.sh" "$ARCHIVE_SRC" "$SCENES_PATH"

# With STANDARD_E2E_DEBUG=true only the first scene of the split is processed;
# unset it for the full split. The cameras_identity_adapter's max_size param (in
# the config) downscales each 3504x2272 ring frame to keep the per-frame .npz
# manageable across the 6-camera rig.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset kitscenes_multimodal \
    --input_path=$SCENES_PATH \
    --output_path=$OUTPUT_PATH \
    --split=$SPLIT \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
