#!/usr/bin/env bash
#
# Preprocess Waymo E2E and Waymo Perception (training + val) into the
# unified StandardE2E format. Run from the repo root.
#
# Override any of the paths / settings via environment variables, e.g.:
#
#   OUTPUT_PATH=/data/processed CONFIG_FILE=configs/base.yaml \
#       bash scripts/prepare_datasets_waymo_e2e_perception.sh
#
# For a smoke run on the first tfrecord of each split, prefix with
# STANDARD_E2E_DEBUG=true.

set -euo pipefail

E2E_INPUT="${E2E_INPUT:-/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0}"
PERC_INPUT="${PERC_INPUT:-/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3/individual_files}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/nvme1/data/standard_e2e}"
NUM_WORKERS="${NUM_WORKERS:-32}"
CONFIG_FILE="${CONFIG_FILE:-configs/base.yaml}"

run() {
    uv run python -m standard_e2e.caching.process_source_dataset "$@" \
        --output_path="$OUTPUT_PATH" \
        --num_workers="$NUM_WORKERS" \
        --config_file="$CONFIG_FILE" \
        --do_parallel_processing
}

run waymo_e2e        --input_path="$E2E_INPUT"  --split=training
run waymo_e2e        --input_path="$E2E_INPUT"  --split=val

run waymo_perception --input_path="$PERC_INPUT" --split=training
run waymo_perception --input_path="$PERC_INPUT" --split=validation
