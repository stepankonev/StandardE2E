# For processing the Waymo E2E dataset with default settings,
# you can use the following script.
# python -m standard_e2e.caching.process_source_dataset waymo_e2e \
#     --input_path=path/to/input \
#     --output_path=path/to/output \
#     --split=training \
#     --num_workers=32 \
#     --config_file=path/to/config.yaml \
#     --do_parallel_processing

# If more customization is needed, you can utilize the script below as example.

import argparse

from standard_e2e.caching.adapters import (
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PanoImageAdapter,
    PastStatesIdentityAdapter,
)
from standard_e2e.caching.src_datasets.waymo_e2e import (
    WaymoE2EDatasetConverter,
    WaymoE2EDatasetProcessor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Waymo E2E dataset processing")
    parser.add_argument(
        "--e2e_dataset_path",
        type=str,
        help="Path to the original E2E dataset \
            (waymo_open_dataset_end_to_end_camera_v_1_0_0)",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to process (training, val, test)",
        choices=["training", "val", "test"],
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        help="Path to the processed data",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.e2e_dataset_path
    output_path = args.processed_data_path
    split = args.split

    processor = WaymoE2EDatasetProcessor(
        common_output_path=output_path,
        split=split,
        adapters=[
            PanoImageAdapter(max_size=384),
            IntentIdentityAdapter(),
            PastStatesIdentityAdapter(),
            FutureStatesIdentityAdapter(),
        ],
    )
    converter = WaymoE2EDatasetConverter(
        source_processor=processor,
        input_path=input_path,
        split=split,
        num_workers=32,
        do_parallel_processing=False,
    )
    # .convert() will process the dataset and save the output in the specified format
    # and produce index file in the output directory.
    converter.convert()


if __name__ == "__main__":
    main()
