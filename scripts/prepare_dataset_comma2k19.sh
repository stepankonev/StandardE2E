#!/usr/bin/env bash
# comma2k19 must be UNZIPPED before processing (as with WayveScenes101):
#   1) extract the distributed Chunk_*.zip archives once into EXTRACTED_PATH;
#   2) point --input_path at EXTRACTED_PATH -- the processor reads the
#      <dongle>|<route>/<segment>/ directories it contains.

ZIP_DIR="/mnt/bigdisk/comma2k19/comma2k19"
EXTRACTED_PATH="/mnt/nvme1/data/comma2k19_extracted"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_comma2k19"
NUM_WORKERS=8
CONFIG_PATH="/home/stvn/code/StandardE2E/configs/comma2k19.yaml"

# One-time extraction (idempotent: -n skips files that already exist). Each
# chunk is ~9 GB zipped; extract only the chunks you intend to process.
mkdir -p "$EXTRACTED_PATH"
for zip in "$ZIP_DIR"/Chunk_*.zip; do
    echo "extracting $(basename "$zip") ..."
    unzip -n -q "$zip" -d "$EXTRACTED_PATH"
done

# With STANDARD_E2E_DEBUG=true only the first segment is processed (~1200
# frames at 20 Hz, full stride); unset it for the full extracted dataset.
# To cut volume / time: --frame_stride N keeps every Nth frame, and the
# cameras_identity_adapter's max_size param (in the config) downscales each
# frame's longest side to N px (native is 20 Hz, 1164x874).
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset comma2k19 \
    --input_path=$EXTRACTED_PATH \
    --output_path=$OUTPUT_PATH \
    --split=all \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --frame_stride=1 \
    --do_parallel_processing
