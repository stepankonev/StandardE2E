#!/usr/bin/env bash
# nuScenes must be EXTRACTED before processing: the official release ships as tar
# archives (v1.0-mini.tgz, or v1.0-trainval_meta.tgz + the *_blobs.tgz). They are
# extracted into a single DATAROOT (see scripts/extract_nuscenes.sh), then
# --input_path points at it -- the processor reads <DATAROOT>/<version>/*.json
# and the samples/ keyframe sensor files.

ARCHIVE_SRC="/mnt/bigdisk/nusc"             # dir holding v1.0-mini.tgz (etc.)
DATAROOT="/mnt/nvme1/data/nuscenes_mini"    # extracted root
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_nuscenes"
# Official nuScenes split; also selects the metadata version (mini_* -> v1.0-mini,
# train|val -> v1.0-trainval, test -> v1.0-test).
SPLIT="mini_val"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/nuscenes.yaml"

# One-time extraction (idempotent: --skip-old-files keeps already-extracted ones).
bash "$SCRIPT_DIR/extract_nuscenes.sh" "$ARCHIVE_SRC" "$DATAROOT"

# With STANDARD_E2E_DEBUG=true only the first scene of the split is processed;
# unset it for the full split. The cameras_identity_adapter's max_size param (in
# the config) downscales each 1600x900 frame to keep the per-frame .npz
# manageable across the 6-camera rig.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset nuscenes \
    --input_path=$DATAROOT \
    --output_path=$OUTPUT_PATH \
    --split=$SPLIT \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
