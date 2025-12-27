# Adding new datasets to `StandardE2E`
`StandardE2E` provides a unified API for working with various datasets for end-to-end self-driving. Thus, it is all based on the original datasets and the ability to add new datasets is the key feature to expand `StandardE2E` functionality.

Community's contribution to adding new datasets would be highly appreciated.

## Files structure
Logic for processing original datasets is contained in `standard_e2e/caching/src_datasets`

```
    standard_e2e/caching/src_datasets
    ├─ __init__.py
    ├─ source_dataset_converter.py
    ├─ source_dataset_processor.py
    ├─ adding_new_dataset.md
    ├─ waymo_e2e
    │  ├─ __init__.py
    │  ├─ waymo_e2e_dataset_converter.py
    │  └─ waymo_e2e_dataset_processor.py
    ├─ waymo_perception
    │  ├─ __init__.py
    │  ├─ waymo_perception_dataset_converter.py
    │  └─ waymo_perception_dataset_processor.py
    ... (other datasets can live alongside these)
    └─ some_new_dataset
    ├─ __init__.py
    ├─ new_dataset_dataset_converter.py
    └─ new_dataset_dataset_processor.py
```

### Dataset Processor
[`SourceDatasetProcessor`](../source_dataset_processor.py) generally holds the logic of converting original dataset frame into `TransformedFrameData` through `StandardFrameData` and generating `FrameIndexData`. While `StandardFrameData` to `TransformedFrameData` transformation is implemented within the final `process_frame()` frame, user needs to implement `raw_frame_data: Any` to `StandardFrameData` logic in `_prepare_standardized_frame_data()` method. This is basically the key functionality for dataset processor.

 `SourceDatasetProcessor` also keeps `AbstractAdapter`s and `SegmentContextAggregator`s. We cover `SegmentContextAggregator` usage later in this instruction.

### Dataset Converter
[`SourceDatasetConverter`](../source_dataset_converter.py) holds the general logic for preprocessing dataset. First, user needs to implement `_get_source_dataset_iterator()` which should return the frame-by-frame iterable over the dataset. Preprocessing the main logic is held in `convert()` method which consists of 2 main parts:
1. `_convert_frames()` -> `index_df`

    calls `process_frame_and_save_data` method from [`SourceDatasetProcessor`](../source_dataset_processor.py) that does `raw_frame_data` -> `StandardFrameData` -> `TransformedFrameData` -> `*.npz` file cache dump.
2. `_run_context_aggregators(index_df)`

    runs [`SegmentContextAggregator`](../segment_context/segment_context_aggregator.py)s. The key idea of [`SegmentContextAggregator`](../segment_context/segment_context_aggregator.py) is to collect data from neighbouring frames within the same segment into a given frame. One example where it is required is collecting thajectory data for a given frame in `Waymo Perception` dataset. While `Waymo E2E` dataset frame data contains future and past coordinates within a single frame, `Waymo Perception` frame only has the current position. Thus, in order to have past and future coordinates for `Waymo Perception` frame we need to do segment context aggregation with [`FuturePastStatesFromMatricesAggregator`](../caching/segment_context/future_past_states_from_matrices.py).
