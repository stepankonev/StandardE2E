# NAVSIM

[NAVSIM](https://github.com/autonomousvision/navsim) builds on OpenScene-v1.1, a
redistribution of nuPlan: per-log pickles of frame dicts, a `sensor_blobs` tree with
eight camera JPEGs and a merged lidar point cloud per frame, and the nuPlan HD maps.
Only the `trainval` split is published; train/val/test subsets are sampled from it
(NAVSIM's `SceneFilter`, or filters on the produced `index.parquet`).

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 8 (front, 3 left, 3 right, rear) | merged sweep | ✓ | ✓ | ✓ | — |

## What StandardE2E extracts

- **One frame = one ego timestamp of a log pickle**; **one segment = one log**.
- **cameras**: the eight cameras with their intrinsics, distortion and
  sensor-to-lidar extrinsics.
- **lidar_pc**: the merged sweep (xyz) from the frame's `.pcd`, in the ego frame.
- **detections_3d**: the annotated boxes, each nuPlan class folded into the coarse
  `DetectionType`.
- **intent**: the four-way one-hot `driving_command`.
- **hd_map**: queried around the ego through nuplan-devkit's map API
  (`maps/<city>/<version>/map.gpkg`) and translated to the unified `MapElementType`
  taxonomy (`_navsim_map.py` has the per-layer rules). nuPlan stores no paint type, so
  lane boundaries carry none.
- **past/future ego trajectories and future detections**: from the per-frame poses and
  boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator`.

StandardE2E reads the pickles, JPEGs and PCDs itself. The `navsim` package is not a
dependency: it pins NumPy 1.23 and PyTorch 2.0, which have no Python 3.12 wheels.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and [nuplan-devkit](https://github.com/motional/nuplan-devkit),
installed from Motional's repository at the `nuplan-devkit-v1.2` tag, as its
[installation guide](https://nuplan-devkit.readthedocs.io/en/latest/installation.html)
describes. Motional does not publish the devkit on PyPI; the `nuplan-devkit` project
there is a third-party upload (version 2.0.0 runs an install-time hook that installs
another repository), so never depend on it. The devkit declares no dependencies, so the
map-stack packages it imports (GeoPandas, Rasterio, Shapely, boto3, ...) are listed
here. StandardE2E's training-only dependencies (PyTorch) are left out: preprocessing
never imports them. [`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/navsim <command>`.

## Pipeline

Download OpenScene-v1.1 and the nuPlan maps with NAVSIM's scripts (see its
[installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)),
set the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/navsim/README.md
```

`debug = true` processes only the first log, as a smoke test.

```stepix
[params]
input_path = "/path/to/openscene-v1.1"      # holds navsim_logs/, sensor_blobs/, maps/
maps_mirror = "/path/to/navsim_maps_rw"     # writable folder, see step 2
output_path = "/path/to/standard_e2e_data"
config = "../../configs/base.yaml"
num_workers = 8
debug = false
```

### 1. Check the input

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.input_path}/navsim_logs/trainval -maxdepth 1 -name '*.pkl' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No *.pkl logs in" {params.input_path}/navsim_logs/trainval >&2
    exit 1
fi
for p in sensor_blobs/trainval maps/nuplan-maps-v1.0.json; do
    if [ ! -e {params.input_path}/"$p" ]; then
        echo "Missing" {params.input_path}/"$p" >&2
        exit 1
    fi
done
echo "Found $n trainval log(s)."
'''
cache = true
```

### 2. Mirror the maps

nuplan-devkit writes a `.maplocks/` folder into the maps root, which fails on a
read-only dataset mount. [`maps_mirror.sh`](maps_mirror.sh) builds a writable mirror in
`maps_mirror`: a copy of the JSON manifest plus one symlink per city, so no map files
are copied.

```stepix
[steps.maps]
env = "."
run = "bash maps_mirror.sh {params.input_path}/maps {params.maps_mirror}"
requires = ["check"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/navsim/trainval/` (one `.npz` per frame plus `index.parquet`),
reading the HD maps through the mirror (`NUPLAN_MAPS_ROOT`).

```stepix
[steps.preprocess]
env = "."
run = '''
NUPLAN_MAPS_ROOT={params.maps_mirror} STANDARD_E2E_DEBUG={params.debug} \
python -m standard_e2e.caching.process_source_dataset navsim \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=trainval \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["maps"]
cache = true
```
