# For processing the Waymo Perception dataset with default settings,
# you can use the following script.
# python -m standard_e2e.caching.process_source_dataset waymo_perception \
#     --input_path=path/to/waymo_open_dataset_v_1_4_3/individual_files \
#     --output_path=path/to/output \
#     --split=validation \
#     --num_workers=32 \
#     --config_file=configs/perception.yaml \
#     --do_parallel_processing

# If more customization is needed, you can utilize the script below as example.
# Unlike the Waymo E2E example, this one materializes the new modalities
# introduced for perception support: LIDAR_PC, HD_MAP, and DETECTIONS_3D.

import argparse

from standard_e2e.caching.adapters import (
    Detections3DIdentityAdapter,
    HDMapIdentityAdapter,
    LidarPCIdentityAdapter,
    PanoImageAdapter,
)
from standard_e2e.caching.segment_context import (
    FutureDetectionsAggregator,
    FuturePastStatesFromMatricesAggregator,
)
from standard_e2e.caching.src_datasets.waymo_perception import (
    WaymoPerceptionDatasetConverter,
    WaymoPerceptionDatasetProcessor,
)
from standard_e2e.caching.src_datasets.waymo_perception.hd_map_ego_crop import (
    WaymoHDMapEgoCropAggregator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Waymo Perception dataset processing")
    parser.add_argument(
        "--perception_dataset_path",
        type=str,
        help="Path to the original Perception dataset \
            (waymo_open_dataset_v_1_4_3/individual_files)",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to process (training, validation, testing)",
        choices=[
            "training",
            "validation",
            "testing",
            "testing_3d_camera_only_detection",
            "domain_adaptation",
        ],
        required=True,
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        help="Path to the processed data",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Worker processes for the per-frame parallel pass.",
    )
    parser.add_argument(
        "--hd_map_crop_extent_m",
        type=float,
        default=WaymoPerceptionDatasetProcessor.DEFAULT_HD_MAP_CROP_EXTENT_M,
        help="Half-extent (metres) for the per-frame ego-relative HD-map crop.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.perception_dataset_path
    output_path = args.processed_data_path
    split = args.split
    source_data_path = f"{input_path}/{split}"

    processor = WaymoPerceptionDatasetProcessor(
        common_output_path=output_path,
        split=split,
        source_data_path=source_data_path,
        hd_map_crop_extent_m=args.hd_map_crop_extent_m,
        adapters=[
            PanoImageAdapter(),
            Detections3DIdentityAdapter(),
            LidarPCIdentityAdapter(),
            HDMapIdentityAdapter(),
        ],
        context_aggregators=[
            FuturePastStatesFromMatricesAggregator(output_path),
            FutureDetectionsAggregator(output_path),
            WaymoHDMapEgoCropAggregator(
                data_path=output_path,
                source_data_path=source_data_path,
                x_range=args.hd_map_crop_extent_m,
                y_range=args.hd_map_crop_extent_m,
            ),
        ],
    )
    converter = WaymoPerceptionDatasetConverter(
        source_processor=processor,
        input_path=input_path,
        split=split,
        num_workers=args.num_workers,
        do_parallel_processing=True,
    )
    # .convert() runs the per-frame parallel pass and then sequentially runs
    # the configured context aggregators (future-window detection aggregation
    # and the HD-map world->ego crop) before writing index.parquet.
    converter.convert()


if __name__ == "__main__":
    main()
