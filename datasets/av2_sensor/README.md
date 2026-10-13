# Argoverse 2 Sensor

The [Argoverse 2 Sensor dataset](https://www.argoverse.org/av2.html#sensor-link) has
1000 ~15 s logs from six US cities: seven ring cameras, two stereo cameras, two roof
lidars merged into one 10 Hz sweep, 3D cuboid annotations and a per-log vector HD map.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 7 ring cameras | merged sweep | ✓ | ✓ | — | — |

## What StandardE2E extracts

- **One frame = one lidar sweep timestamp**; **one segment = one log**.
- **cameras**: the seven ring cameras, each matched to the sweep by nearest timestamp.
  The stereo pair is left out: it runs at 5 Hz and does not line up 1:1 with the sweeps.
- **lidar_pc**: the merged sweep (xyz) in the ego frame, read with av2's
  `Sweep.from_feather`.
- **detections_3d**: the cuboids annotated for that exact sweep, in the ego frame.
- **hd_map**: the log's vector map, loaded with av2's `ArgoverseStaticMap` (lane
  segments with their boundaries and paint, drivable areas, pedestrian crossings) and
  translated to the unified `MapElementType` taxonomy. Map geometry is moved into the
  ego frame in 2D (see [Known issues](#known-issues)).
- **past/future ego trajectories and future detections**: from the per-sweep poses and
  boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator`.

## Splits

The dataset ships three splits, one folder each under `input_path`.

| Split | What it holds | Default |
|---|---|---|
| `train` | 700 logs with 3D cuboid annotations | on |
| `val` | 150 logs with 3D cuboid annotations | on |
| `test` | 150 logs; annotations withheld, so the frames carry no boxes | on |

## Known issues

- **NaN heights in the vector maps** (source data). Some logs' map JSON stores NaN
  heights for part of the geometry, so StandardE2E moves map geometry into the ego frame
  in 2D, as av2's own rasteriser does: map elements carry no height.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and the [av2](https://github.com/argoverse/av2-api) SDK. av2
requires the full `opencv-python`, which would install over StandardE2E's
`opencv-python-headless` (both ship the same `cv2/` package); an override keeps only the
headless build. av2 itself requires PyTorch, so of StandardE2E's training-only
dependencies only torchvision is left out. [`.python-version`](.python-version)
pins Python 3.12, the newest version av2 0.3.6 ships wheels for. stepix builds the environment into `.venv/` here on
first use; to run a command in it yourself:
`uv run --project datasets/av2_sensor <command>`.

## Pipeline

Download the sensor dataset (see the
[Argoverse User Guide](https://argoverse.github.io/user-guide/datasets/sensor.html)),
set the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/av2_sensor/README.md
```

`debug = true` processes only the first log of each split, as a smoke test.
`download = true` copies the enabled splits from the public Argoverse bucket first (step 1).

```stepix
[params]
input_path = "/path/to/argoverse2/sensor"   # holds train/, val/, test/
output_path = "/path/to/standard_e2e_data"
config = "../../configs/base.yaml"
num_workers = 8
download = false                            # true: fetch the data first (step 1)
debug = false
# Splits to preprocess, one step each (see Splits):
train = true
val = true
test = true
```

### 1. Download (optional)

With `download = true`, this step copies the enabled splits' logs from the public
Argoverse bucket into `input_path/<split>/` with [s5cmd](https://github.com/peak/s5cmd)
(run through `uvx`), as the
[Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html)
suggests, skipping files already there. No account is needed.

```stepix
[steps.download]
env = "."
run = '''
if [ {params.download} != true ]; then
    echo "download = false: nothing to download."
    exit 0
fi
splits=""
if [ {params.train} = true ]; then splits="$splits train"; fi
if [ {params.val} = true ]; then splits="$splits val"; fi
if [ {params.test} = true ]; then splits="$splits test"; fi
if [ -z "$splits" ]; then
    echo "No split is set to true: nothing to download."
    exit 0
fi
for split in $splits; do
    uvx s5cmd --no-sign-request cp --no-clobber \
        "s3://argoverse/datasets/av2/sensor/$split/*" {params.input_path}/"$split"/ || exit 1
done
'''
cache = true
```

### 2. Check the input

Each enabled split folder holds one directory per log, with its sweeps in
`sensors/lidar/` (only the first log is inspected, so the check stays quick).

```stepix
[steps.check]
env = "."
run = '''
splits=""
if [ {params.train} = true ]; then splits="$splits train"; fi
if [ {params.val} = true ]; then splits="$splits val"; fi
if [ {params.test} = true ]; then splits="$splits test"; fi
if [ -z "$splits" ]; then
    echo "No split is set to true." >&2
    exit 1
fi
for split in $splits; do
    logs={params.input_path}/"$split"
    n=$(find "$logs" -mindepth 1 -maxdepth 1 -type d | wc -l)
    first=$(find "$logs" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
    if [ "$n" -eq 0 ] || [ ! -d "$first/sensors/lidar" ]; then
        echo "No <log>/sensors/lidar/ directories in" "$logs" >&2
        exit 1
    fi
    echo "Found $n $split log(s)."
done
'''
requires = ["download"]
cache = true
```

### 3. Preprocess

One step per split: each writes `<output_path>/av2_sensor/<split>/` (one `.npz` per frame
plus `index.parquet`) when its split's param is `true`. stepix caches each split's step
on its own, so turning a split on or off, or a failure in one, leaves the others alone.

```stepix
[steps.preprocess_train]
env = "."
run = '''
if [ {params.train} != true ]; then
    echo "train = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset av2_sensor \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=train \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_val]
env = "."
run = '''
if [ {params.val} != true ]; then
    echo "val = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset av2_sensor \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=val \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_test]
env = "."
run = '''
if [ {params.test} != true ]; then
    echo "test = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset av2_sensor \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=test \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```
