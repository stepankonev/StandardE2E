#!/usr/bin/env bash
INPUT_PATH="/mnt/bigdisk/datasets/argoverse2/sensor"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_av2"
NUM_WORKERS=4
CONFIG_PATH="/home/stvn/code/StandardE2E/configs/base.yaml"

# AV2 sensor splits are train / val / test. With STANDARD_E2E_DEBUG=true
# only the first log of each split is processed (~1 GB output / split);
# unset for full preprocessing.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset av2_sensor \
    --input_path=$INPUT_PATH \
    --output_path=$OUTPUT_PATH \
    --split=train \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing

STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset av2_sensor \
    --input_path=$INPUT_PATH \
    --output_path=$OUTPUT_PATH \
    --split=val \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
