#!/usr/bin/env bash
# KITScenes *LongTail* (dataset key `kitscenes_longtail`; the long-tail /
# reasoning-traces sibling of `kitscenes_multimodal`, arXiv 2603.23607). Unlike
# Multimodal, LongTail ships as Hugging Face `datasets` parquet -- no extraction
# step, just download the shards.
#
# 1. Download a split from Hugging Face (gated; accept the terms first), e.g.:
#      hf download KIT-MRT/KITScenes-LongTail --repo-type dataset \
#          --include "data/train-*" --local-dir "$DATA_ROOT"
#    Splits: train / test / train_raw / test_raw (val + full train ship later).
#    "raw" = native-resolution frames; the non-raw splits are the processed
#    frames the vendored camera intrinsics match. The `test` split withholds the
#    future trajectories (eval ground truth); only train/train_raw carry real
#    expert + counterfactual futures.
# 2. --input_path points at the dataset root (the folder holding data/) or at
#    the data/ folder itself; the processor reads data/<split>-*.parquet.

DATA_ROOT="/mnt/bigdisk/KITScenes-LongTail"   # holds data/<split>-*.parquet
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_kitscenes_longtail"
# train / test / train_raw / test_raw / val.
SPLIT="train"
NUM_WORKERS=4
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/kitscenes_longtail.yaml"

# With STANDARD_E2E_DEBUG=true only the first scenario of the split is processed;
# unset it for the full split. The cameras_identity_adapter's max_size param (in
# the config) downscales each 3504x2272 ring frame across the 6-camera rig.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path=$DATA_ROOT \
    --output_path=$OUTPUT_PATH \
    --split=$SPLIT \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
