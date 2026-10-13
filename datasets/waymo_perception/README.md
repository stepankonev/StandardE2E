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
  `SIDE_RIGHT`) with intrinsics and distortion from the calibration, and
  `T_ego_from_camera` extrinsics converted from Waymo's x-forward camera frame to the
  OpenCV optical frame (x right, y down, z forward).
- **lidar_pc**: the first return of all five lidars in the vehicle frame. The range
  images are decoded by `standard_e2e/utils/waymo_lidar_numpy.py`, a NumPy port of the
  SDK's `frame_utils` decode kept because it is about 2.7× faster per frame than the
  TensorFlow ops; `tests/test_waymo_lidar_numpy.py` checks it against the SDK version.
- **detections_3d**: the 3D box labels, in the vehicle frame.
- **hd_map**: Waymo puts `map_features` on the first frame of each segment only. The
  converter pre-scans every segment's first record before the worker pool starts,
  caches the world-frame features (in `/dev/shm` when it has 500 MB free, else the
  output directory or the system temp directory), and every frame applies its inverse
  pose, with the frame's `map_pose_offset` added to the pose's translation first, as the
  SDK's maps tutorial does. The offset is Waymo's correction for the pose drifting
  against the map along a segment: about zero on the first frame, it reached 0.55 m
  horizontally and 0.81 m vertically in 17 sampled v1.4.3 segments.
- **past/future ego trajectories and future detections**: from the per-frame poses and
  boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator` after all frames are written.

## Splits

v1.4.3's `individual_files/` holds one folder of `*.tfrecord` segments per split.

| Split | What it holds | Default |
|---|---|---|
| `training` | 798 segments with 3D box labels | on |
| `validation` | 202 segments with 3D box labels | on |
| `testing` | 150 segments; labels withheld | on |

Two more folders are not covered: `testing_3d_camera_only_detection/` (80 segments of
the camera-only detection challenge) has no lidar range images, which the lidar decode
fails on, and `domain_adaptation/` nests its own `training/`, `testing/` and
`testing_location/` folders, which the converter does not look into.

## Known issues

- **`FRONT_RIGHT` and `SIDE_LEFT` have each other's calibration** (StandardE2E bug, not
  fixed yet). The frames list their images as `FRONT`, `FRONT_LEFT`, `SIDE_LEFT`,
  `FRONT_RIGHT`, `SIDE_RIGHT` but their calibrations in name order, and StandardE2E pairs
  the two lists by position, so these two cameras get each other's intrinsics,
  extrinsics and distortion. Waymo Open E2E lists both in the same order and is not
  affected.

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

`debug = true` processes only the first segment of each split, as a smoke test.
`download = true` copies the enabled splits from Waymo's bucket first (step 1), with your
Google account.

```stepix
[params]
input_path = "/path/to/waymo_open_dataset_v_1_4_3/individual_files"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/base.yaml"
num_workers = 8
download = false                        # true: fetch the data first (step 1)
debug = false
# Splits to preprocess, one step each (see Splits):
training = true
validation = true
testing = true
```

### 1. Download (optional)

With `download = true`, this step copies the enabled splits' folders of v1.4.3's
`individual_files/` from Waymo's Cloud Storage bucket into `input_path` with the
[Google Cloud CLI](https://cloud.google.com/sdk/docs/install), skipping files already
there. It needs `gcloud` logged in (`gcloud auth login`) with a Google account
registered for the [Waymo Open Dataset](https://waymo.com/open/); otherwise the copy
is denied.

```stepix
[steps.download]
env = "."
run = '''
if [ {params.download} != true ]; then
    echo "download = false: nothing to download."
    exit 0
fi
if ! command -v gcloud >/dev/null; then
    echo "download = true needs the Google Cloud CLI: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
fi
splits=""
if [ {params.training} = true ]; then splits="$splits training"; fi
if [ {params.validation} = true ]; then splits="$splits validation"; fi
if [ {params.testing} = true ]; then splits="$splits testing"; fi
if [ -z "$splits" ]; then
    echo "No split is set to true: nothing to download."
    exit 0
fi
mkdir -p {params.input_path}
for split in $splits; do
    gcloud storage cp -r --no-clobber \
        "gs://waymo_open_dataset_v_1_4_3/individual_files/$split" \
        {params.input_path}/ || {
        echo "Copy failed. If access was denied, register this Google account at" >&2
        echo "https://waymo.com/open/ and log in with 'gcloud auth login'." >&2
        exit 1
    }
done
'''
cache = true
```

### 2. Check the input

Each enabled split is a folder of `*.tfrecord` segments inside `input_path`.

```stepix
[steps.check]
env = "."
run = '''
splits=""
if [ {params.training} = true ]; then splits="$splits training"; fi
if [ {params.validation} = true ]; then splits="$splits validation"; fi
if [ {params.testing} = true ]; then splits="$splits testing"; fi
if [ -z "$splits" ]; then
    echo "No split is set to true." >&2
    exit 1
fi
for split in $splits; do
    n=$(find {params.input_path}/"$split" -maxdepth 1 -name '*.tfrecord' | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "No *.tfrecord files in" {params.input_path}/"$split" >&2
        exit 1
    fi
    echo "Found $n $split segment(s)."
done
'''
requires = ["download"]
cache = true
```

### 3. Preprocess

One step per split: each writes `<output_path>/waymo_perception/<split>/` (one `.npz` per frame
plus `index.parquet`) when its split's param is `true`. stepix caches each split's step
on its own, so turning a split on or off, or a failure in one, leaves the others alone.

```stepix
[steps.preprocess_training]
env = "."
run = '''
if [ {params.training} != true ]; then
    echo "training = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=training \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_validation]
env = "."
run = '''
if [ {params.validation} != true ]; then
    echo "validation = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=validation \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_testing]
env = "."
run = '''
if [ {params.testing} != true ]; then
    echo "testing = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset waymo_perception \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=testing \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```
