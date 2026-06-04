#!/usr/bin/env bash
INPUT_PATH="/mnt/bigdisk/comma2k19/comma2k19"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_comma2k19"
NUM_WORKERS=8
CONFIG_PATH="/home/stvn/code/StandardE2E/configs/comma2k19.yaml"

# comma2k19 reads either extracted segment directories or the distributed
# Chunk_*.zip archives directly (each segment's video.hevc is extracted to a
# scratch file under the output dir and cleaned up afterwards). With
# STANDARD_E2E_DEBUG=true only the first segment is processed (~1200 frames at
# 20 Hz, full stride); unset it for the full ~2000-segment dataset. Native
# rate is 20 Hz -- raise --frame_stride to subsample and cut output volume.
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset comma2k19 \
    --input_path=$INPUT_PATH \
    --output_path=$OUTPUT_PATH \
    --split=all \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --frame_stride=1 \
    --do_parallel_processing
