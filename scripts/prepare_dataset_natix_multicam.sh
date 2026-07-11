#!/usr/bin/env bash
# NATIX Multi-Camera Driving Dataset (dataset key `natix_multicam`). No
# extraction step -- the download is a plain directory tree
# (<Country>/[<State>/]<trip>/<HH-MM-SS>/<CAMERA>_FOLDER/*.{mp4,csv}).
#
# 1. Download the data (the Hugging Face repo is gated; accept the terms
#    first). The repo itself carries a ~20-minute sample:
#      hf download natix-network-org/natix-multi-camera-driving-dataset \
#          --repo-type dataset --local-dir "$DATA_ROOT"
#    The full 100 h release (~1.28 TB) is pulled from a Cloudflare R2 bucket
#    with boto3 credentials provided on access approval -- see the download
#    snippet on the dataset card.
# 2. --input_path points at the downloaded root (or its dataset-sample/
#    folder); trips are discovered by their fixed_metadata.json marker.

DATA_ROOT="/mnt/storage/data/natix/demo-natix-multi-camera-driving-dataset"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_natix_multicam"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/natix_multicam.yaml"

# With STANDARD_E2E_DEBUG=true only the first trip is processed; unset it for
# the full download. One frame is emitted per front-camera GPS fix (~1-10 Hz;
# the ~36 fps video between fixes has no pose) -- use --frame_stride N to
# subsample further, and the cameras_identity_adapter's max_size param (in
# the config) to downscale the up-to-six cameras per frame.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset natix_multicam \
    --input_path="$DATA_ROOT" \
    --output_path=$OUTPUT_PATH \
    --split=all \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --frame_stride=1 \
    --do_parallel_processing
