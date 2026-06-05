#!/usr/bin/env bash
# TruckDrive must be UNZIPPED before processing: the per-scene modality zips are
# expanded into the devkit layout (see scripts/extract_truckdrive.sh), then
# --input_path points at the extracted root -- the processor reads the
# scene_*/{calibrations,camera,lidar,annotations,poses}/ directories.

ZIP_ROOT="/mnt/bigdisk/torc/TruckDrive"
EXTRACTED_PATH="/mnt/nvme1/data/truckdrive_extracted"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_truckdrive"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/truckdrive.yaml"

# One-time extraction (idempotent: unzip -n skips files that already exist).
bash "$SCRIPT_DIR/extract_truckdrive.sh" "$ZIP_ROOT" "$EXTRACTED_PATH"

# With STANDARD_E2E_DEBUG=true only the first scene is processed (~260 frames);
# unset it for the full extracted dataset. The cameras_identity_adapter's
# max_size param (in the config) downscales each 8 MP frame's longest side to
# keep the per-frame .npz manageable across the 11-camera rig.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset truckdrive \
    --input_path=$EXTRACTED_PATH \
    --output_path=$OUTPUT_PATH \
    --split=all \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
