INPUT_PATH_E2E="/mnt/bigdisk/datasets/waymo/waymo_open_dataset_end_to_end_camera_v_1_0_0"
INPUT_PATH_PERC="/mnt/bigdisk/datasets/waymo/waymo_open_dataset_v_1_4_3/individual_files"
OUTPUT_PATH="/mnt/nvme1/data/standard_e2e_test_waymo"
NUM_WORKERS=4
CONFIG_PATH="/home/stvn/code/StandardE2E/configs/base.yaml"


STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset waymo_e2e \
    --input_path=$INPUT_PATH_E2E \
    --output_path=$OUTPUT_PATH \
    --split=training \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing

STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset waymo_e2e \
    --input_path=$INPUT_PATH_E2E \
    --output_path=$OUTPUT_PATH \
    --split=val \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing


STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path=$INPUT_PATH_PERC \
    --output_path=$OUTPUT_PATH \
    --split=training \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing

STANDARD_E2E_DEBUG=true uv run python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path=$INPUT_PATH_PERC \
    --output_path=$OUTPUT_PATH \
    --split=validation \
    --num_workers=$NUM_WORKERS \
    --config_file=$CONFIG_PATH \
    --do_parallel_processing
