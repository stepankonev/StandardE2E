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
  ego frame in 2D, because some logs store NaN heights in the map JSON.
- **past/future ego trajectories and future detections**: from the per-sweep poses and
  boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator`.

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

`debug = true` processes only the first log, as a smoke test.

```stepix
[params]
input_path = "/path/to/argoverse2/sensor"   # holds train/, val/, test/
output_path = "/path/to/standard_e2e_data"
split = "train"                              # train, val or test
config = "../../configs/base.yaml"
num_workers = 8
debug = false
```

### 1. Check the input

Each split folder holds one directory per log, with its sweeps in `sensors/lidar/`
(only the first log is inspected, so the check stays quick on a full split).

```stepix
[steps.check]
env = "."
run = '''
logs={params.input_path}/{params.split}
n=$(find "$logs" -mindepth 1 -maxdepth 1 -type d | wc -l)
first=$(find "$logs" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
if [ "$n" -eq 0 ] || [ ! -d "$first/sensors/lidar" ]; then
    echo "No <log>/sensors/lidar/ directories in" "$logs" >&2
    exit 1
fi
echo "Found $n {params.split} log(s)."
'''
cache = true
```

### 2. Preprocess

Writes `<output_path>/av2_sensor/<split>/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset av2_sensor \
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
