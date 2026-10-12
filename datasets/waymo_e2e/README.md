# Waymo Open E2E

The [Waymo Open Dataset End-to-End Driving](https://waymo.com/open/data/e2e/) release
(`waymo_open_dataset_end_to_end_camera_v_1_0_0`) pairs an 8-camera surround rig with
the ego past and future states, a high-level driving command and, on the validation
split, rater-scored preference trajectories. It ships as sharded TFRecords of
`E2EDFrame` protos.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 8 surround | — | — | — | ✓ | ✓ |

## What StandardE2E extracts

- **One frame = one `E2EDFrame`**; **one segment = one run** (the part of the context
  name before `-`, the frame id after it). Frame timestamps are `frame_id × 0.1 s`.
- **cameras**: the eight cameras (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `SIDE_LEFT`,
  `SIDE_RIGHT`, `REAR_LEFT`, `REAR`, `REAR_RIGHT`) with intrinsics, extrinsics and
  distortion from the frame's calibration.
- **past_states / future_states**: the ego trajectories shipped inside each frame
  (positions, plus velocity and acceleration for the past), at 0.25 s steps.
- **intent**: the driving command.
- **preference_trajectory**: the rater-scored trajectories with their scores, where the
  frame has any (scores sum to more than zero).

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
`uv run --project datasets/waymo_e2e <command>`.

## Pipeline

Download the release from the [Waymo Open Dataset](https://waymo.com/open/download/)
page, set the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/waymo_e2e/README.md
```

`debug = true` processes only the first TFRecord file, as a smoke test (it cuts runs
at the file boundary, so use it only to check the setup).

```stepix
[params]
input_path = "/path/to/waymo_open_dataset_end_to_end_camera_v_1_0_0"
output_path = "/path/to/standard_e2e_data"
split = "training"                      # training, val or test
config = "../../configs/base.yaml"
num_workers = 8
debug = false
```

### 1. Check the input

The split's shards are the `<split>*.tfrecord*` files directly inside `input_path`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.input_path} -maxdepth 1 -name {params.split}'*.tfrecord*' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No {params.split}*.tfrecord* files in" {params.input_path} >&2
    exit 1
fi
echo "Found $n {params.split} TFRecord file(s)."
'''
cache = true
```

### 2. Preprocess

Writes `<output_path>/waymo_e2e/<split>/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset waymo_e2e \
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

The converter also accepts `--n_shards N --shard_id I` to process a 1/N slice of the
records, for spreading a split over several machines.
