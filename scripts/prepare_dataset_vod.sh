#!/usr/bin/env bash
# View-of-Delft must be UNZIPPED before processing: scripts/extract_vod.sh unpacks
# the lidar/ tree into the devkit layout, then --input_path points at the
# extracted view_of_delft_PUBLIC -- the processor reads
# lidar/{training,testing}/{calib,velodyne,image_2,label_2,pose}/.

ZIP_ROOT="/mnt/bigdisk/VoD"
EXTRACTED_PATH="/mnt/nvme1/data/vod_extracted"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_vod"
SPLIT="train"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/vod.yaml"

# One-time extraction (idempotent: unzip -n skips files that already exist). The
# third arg overlays per-object track ids; drop it to keep the base labels.
bash "$SCRIPT_DIR/extract_vod.sh" \
    "$ZIP_ROOT/view_of_delft_detection_PUBLIC.zip" \
    "$EXTRACTED_PATH" \
    "$ZIP_ROOT/label_2_with_track_ids.zip"

# With STANDARD_E2E_DEBUG=true only the first scene of the split is processed;
# unset it for the full split. Note: --split=test ships sensor data without
# labels (detections will be empty for test frames).
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset vod \
    --input_path="$EXTRACTED_PATH/view_of_delft_PUBLIC" \
    --output_path=$OUTPUT_PATH \
    --split=$SPLIT \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
