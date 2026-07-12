#!/usr/bin/env bash
# NATIX Edge Case Driving Dataset (dataset key `natix_edgecase`) -- the
# curated edge-case sibling of natix_multicam. No extraction step -- the
# download is a plain directory tree in the exact multicam layout
# (data/<Country>/<State>/<trip>/<HH-MM-SS>/<CAMERA>_FOLDER/*.{mp4,csv})
# plus data/edge-case.json (the per-clip VLM event annotations).
#
# 1. Download the data (the Hugging Face repo is gated; accept the terms
#    first). The first release is ~3.3 GB:
#      hf download natix-network-org/natix-edge-case-driving-dataset \
#          --repo-type dataset --local-dir "$DATA_ROOT"
# 2. --input_path points at the downloaded root (or its data/ folder); trips
#    are discovered by their fixed_metadata.json marker and edge-case.json by
#    walking up from each trip.

DATA_ROOT="/mnt/storage/data/natix/natix-edge-case-driving-dataset"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_natix_edgecase"
NUM_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/../configs/natix_edgecase.yaml"

# With STANDARD_E2E_DEBUG=true only the first trip is processed; unset it for
# the full release. One frame is emitted per front-camera GPS fix (~1-10 Hz;
# the ~36 fps video between fixes has no pose) -- use --frame_stride N to
# subsample further, and the cameras_identity_adapter's max_size param (in
# the config) to downscale the up-to-six cameras per frame. Frames covered by
# an annotated event are filterable from the index (extra_edge_case_count >
# 0) and carry the full VLM analysis in aux_data["edge_case_events"].
STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset natix_edgecase \
    --input_path="$DATA_ROOT" \
    --output_path=$OUTPUT_PATH \
    --split=all \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --frame_stride=1 \
    --do_parallel_processing
