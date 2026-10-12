# nuScenes

[nuScenes](https://www.nuscenes.org/nuscenes) (Motional, CVPR 2020) is the de facto
surround-view E2E / BEV benchmark: 1000 ~20 s scenes with a 6-camera surround rig
(1600×900), a 32-beam `LIDAR_TOP` and densely annotated 3D boxes at 2 Hz keyframes, plus
a separate vector HD map ("map expansion").

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 6 surround | LIDAR_TOP | ✓ (map expansion) | ✓ | — | — |

## What StandardE2E extracts

- **One frame = one keyframe sample**; **one segment = one scene**. Everything is in the
  ego frame.
- **cameras**: the six `CAM_*` channels, mapped onto the canonical `CameraDirection`
  members.
- **lidar_pc**: the `LIDAR_TOP` cloud (xyz).
- **detections_3d**: the `sample_annotation` boxes moved from the global frame into the
  ego frame, each `category_name` folded into the coarse `DetectionType` (nuScenes does
  not annotate the ego vehicle, so nothing is excluded).
- **hd_map**: the vector map expansion (lane centers from the arcline paths, lane and
  road dividers, crossings, walkways, stop lines, drivable area, intersections),
  translated to the unified `MapElementType` taxonomy and rasterised by
  `HDMapBEVAdapter`. It needs the separate `nuScenes-map-expansion-v1.3` pack in
  `<dataroot>/maps/` (step 1 unzips it from `archives`); without it the HD map is
  skipped.
- **past/future ego trajectories and future detections**: from the per-sample poses
  and boxes, via the `FuturePastStatesFromMatricesAggregator` and
  `FutureDetectionsAggregator`.

The `split` param is an official nuScenes label that also picks the metadata version:
`mini_train` / `mini_val` → `v1.0-mini`, `train` / `val` → `v1.0-trainval`, `test` →
`v1.0-test` (no annotations). The five radars have no StandardE2E target yet. A
partially downloaded trainval converts cleanly: scenes whose sensor files are not on disk
are skipped.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and the official
[nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), which provides the
split scene lists and the lane-arcline discretization, without StandardE2E's
training-only dependencies (PyTorch), which preprocessing never imports. The devkit
pins `numpy<2`, so this environment runs NumPy 1.26. The JSON tables are read by
StandardE2E itself.
[`.python-version`](.python-version) pins Python 3.12, the newest version NumPy 1.26
ships wheels for. stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/nuscenes <command>`.

## Pipeline

Download the release archives (`v1.0-mini.tgz`, or `v1.0-trainval_meta.tgz` and the
`v1.0-trainval*_blobs.tgz` parts) and `nuScenes-map-expansion-v1.3.zip` from the
[nuScenes download page](https://www.nuscenes.org/download), set the paths
below, then run, from the repository root:

```bash
uvx stepix run datasets/nuscenes/README.md
```

`debug = true` processes only the first scene of the split, as a smoke test.

```stepix
[params]
archives = "/path/to/nuscenes_archives"     # "" when dataroot is already extracted
dataroot = "/path/to/nuscenes"              # extracted dataset (written by step 1)
output_path = "/path/to/standard_e2e_data"
split = "mini_val"                          # mini_train, mini_val, train, val or test
config = "../../configs/nuscenes.yaml"
num_workers = 8
debug = false
```

### 1. Extract the archives

[`extract.sh`](extract.sh) extracts every archive into `dataroot` (existing files are
kept, so re-runs are cheap) and unzips the map expansion into `<dataroot>/maps/`. The
extracted copy is about as large as the download; write it somewhere other than a
read-only dataset mount. Set `archives = ""` if `dataroot` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.archives} {params.dataroot}"
cache = true
```

### 2. Check the input

```stepix
[steps.check]
env = "."
run = '''
case {params.split} in
    mini_*) version=v1.0-mini ;;
    test) version=v1.0-test ;;
    *) version=v1.0-trainval ;;
esac
for p in "$version/scene.json" "$version/sample_data.json" samples; do
    if [ ! -e {params.dataroot}/"$p" ]; then
        echo "Missing" {params.dataroot}/"$p" >&2
        exit 1
    fi
done
if [ ! -d {params.dataroot}/maps/expansion ]; then
    echo "No maps/expansion/ in" {params.dataroot} "- the HD map will be skipped."
fi
echo "Found nuScenes $version."
'''
requires = ["extract"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/nuscenes/<split>/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset nuscenes \
    --input_path={params.dataroot} \
    --output_path={params.output_path} \
    --split={params.split} \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
