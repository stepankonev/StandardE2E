# Waymo Open Perception

The [Waymo Open Dataset Perception](https://waymo.com/open/data/perception/) release
(v1.4.3, `individual_files/`) has ~1150 20 s segments at 10 Hz with five cameras, five
lidars, 3D box labels and a per-segment HD map. It ships as one TFRecord of `Frame`
protos per segment.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 5 (front + sides) | top + 4 side lidars | ✓ | ✓ | — | — |

## What StandardE2E extracts

- **One frame = one `Frame` proto**; **one segment = one TFRecord** (a 20 s segment).
- **cameras**: the five cameras (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `SIDE_LEFT`,
  `SIDE_RIGHT`) with intrinsics, extrinsics and distortion from the calibration.
- **lidar_pc**: the first return of all five lidars in the vehicle frame. The range
  images are decoded by `standard_e2e/utils/waymo_lidar_numpy.py`, a NumPy port of the
  SDK's `frame_utils` decode kept because it is about 2.7× faster per frame than the
  TensorFlow ops; `tests/test_waymo_lidar_numpy.py` checks it against the SDK version.
- **detections_3d**: the 3D box labels, in the vehicle frame.
- **hd_map**: Waymo puts `map_features` on the first frame of each segment only. The
  converter pre-scans every segment's first record before the worker pool starts,
  caches the world-frame features (in `/dev/shm` when it has 500 MB free, else the
  output directory or the system temp directory), and every frame applies its inverse
  pose.
- **past/future ego trajectories and future detections**: from the per-frame poses and
  boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator` after all frames are written.

## Known issues

- **`FRONT_RIGHT` and `SIDE_LEFT` have each other's calibration** (StandardE2E bug, not
  fixed yet). The frames list their images as `FRONT`, `FRONT_LEFT`, `SIDE_LEFT`,
  `FRONT_RIGHT`, `SIDE_RIGHT` but their calibrations in name order, and StandardE2E pairs
  the two lists by position, so these two cameras get each other's intrinsics,
  extrinsics and distortion. Waymo Open E2E lists both in the same order and is not
  affected.
- **The HD map is not corrected for pose drift** (StandardE2E bug, not fixed yet). Each
  frame carries a `map_pose_offset` that, per Waymo, must be added to align the frame
  with the map features. StandardE2E ignores it, so the ego-frame map is off by up to
  about 0.3 m horizontally and 0.8 m vertically (the offset is zero on a segment's first
  frame and drifts along it).

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and TensorFlow, which reads the TFRecords and decodes the camera
images (`tensorflow-cpu` on Linux x86-64 and Windows, `tensorflow` on macOS and Linux
ARM), without StandardE2E's training-only dependencies (PyTorch), which preprocessing
never imports. The Waymo protos are vendored in `standard_e2e/third_party/`: the official
`waymo-open-dataset` package pins TensorFlow 2.13 and NumPy 1.23 and cannot share an
environment with StandardE2E. [`.python-version`](.python-version) pins Python 3.12
(`requires-python` allows 3.12–3.13). stepix builds the environment into `.venv/` here
on first use; to run a command in it yourself:
`uv run --project datasets/waymo_perception <command>`.

## Pipeline

Download the v1.4.3 `individual_files/` from the
[Waymo Open Dataset](https://waymo.com/open/download/) page, set the paths below, then
run, from the repository root:

```bash
uvx stepix run datasets/waymo_perception/README.md
```

`debug = true` processes only the first segment, as a smoke test.

```stepix
[params]
input_path = "/path/to/waymo_open_dataset_v_1_4_3/individual_files"
output_path = "/path/to/standard_e2e_data"
split = "training"                      # training, validation, testing, ...
config = "../../configs/base.yaml"
num_workers = 8
debug = false
```

### 1. Check the input

Each split is a folder of `*.tfrecord` segments inside `input_path`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.input_path}/{params.split} -maxdepth 1 -name '*.tfrecord' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No *.tfrecord files in" {params.input_path}/{params.split} >&2
    exit 1
fi
echo "Found $n {params.split} segment(s)."
'''
cache = true
```

### 2. Preprocess

Writes `<output_path>/waymo_perception/<split>/` (one `.npz` per frame plus
`index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split={params.split} \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
