# Argoverse 2 Lidar

The [Argoverse 2 Lidar dataset](https://www.argoverse.org/av2.html#lidar-link) has
20 000 unannotated ~30 s logs of merged roof-lidar sweeps, each with its ego poses and a
vector HD map. Its on-disk format is the Sensor dataset's minus the cameras and the
`annotations.feather` file.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| — | merged sweep | ✓ | — | — | — |

## What StandardE2E extracts

The processor subclasses the [Argoverse 2 Sensor](../av2_sensor/README.md) one and
skips its camera and box loading, so frames, poses, lidar and the HD map are built
exactly as there:

- **One frame = one lidar sweep timestamp**; **one segment = one log**.
- **lidar_pc**: the merged sweep (xyz) in the ego frame.
- **hd_map**: the log's vector map, translated to the unified `MapElementType`
  taxonomy.
- **past/future ego trajectories**: from the per-sweep poses, via the
  `FuturePastStatesFromMatricesAggregator`.

The missing cameras and boxes surface as defaults at training time (`ModalityDefaults`),
so AV2 Lidar frames batch together with camera datasets.

## Known issues

- **NaN heights in the vector maps** (source data), handled as for
  [Argoverse 2 Sensor](../av2_sensor/README.md#known-issues): map elements carry no
  height.

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
`uv run --project datasets/av2_lidar <command>`.

## Pipeline

Download the lidar dataset (see the
[Argoverse User Guide](https://argoverse.github.io/user-guide/datasets/lidar.html)), set
the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/av2_lidar/README.md
```

`debug = true` processes only the first log, as a smoke test.

```stepix
[params]
input_path = "/path/to/argoverse2/lidar"    # holds train/, val/, test/
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

Writes `<output_path>/av2_lidar/<split>/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset av2_lidar \
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
