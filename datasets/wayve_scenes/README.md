# WayveScenes101

[WayveScenes101](https://wayve.ai/science/wayvescenes101) (Wayve) is a novel-view
synthesis benchmark: 101 urban and highway scenes, each ~20 s from a five-camera fisheye
rig, with a per-scene COLMAP reconstruction (camera poses plus a sparse SfM point
cloud). It ships no sensor lidar, HD map or boxes.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 5 fisheye (forward + side arc) | SfM point cloud | — | — | — | — |

## What StandardE2E extracts

- **One frame = one `front-forward` image timestamp**; **one segment = one scene**.
- **cameras**: the five fisheye cameras with their COLMAP intrinsics and distortion.
- **lidar_pc**: no sensor lidar exists, so the scene's **COLMAP SfM** point cloud is
  used instead. It is filtered (reprojection error ≤ 6, track length ≥ 2), converted
  from OpenCV to FLU axes, moved into each frame's ego frame and clipped to 50 m, so it
  flows through the lidar adapters. It is photogrammetric (sparse, up to scale), not a
  sensor measurement; a frame's cloud lifted by `aux_data["pose_matrix"]` reproduces the
  source SfM cloud.
- **past/future ego trajectories**: from the COLMAP poses, via the
  `FuturePastStatesFromMatricesAggregator`.

The dataset has no perception split, so `split` is a passthrough label for the output
folder.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable), without its training-only dependencies (PyTorch), which
preprocessing never imports. The COLMAP binaries are parsed by a small reader in
StandardE2E (`_colmap.py`). [`.python-version`](.python-version) pins Python 3.12
(`requires-python` allows 3.12–3.14). stepix builds the environment into `.venv/` here
on first use; to run a command in it yourself:
`uv run --project datasets/wayve_scenes <command>`.

## Pipeline

Download the `scene_*.zip` archives (see the
[WayveScenes101 repository](https://github.com/wayveai/wayve_scenes)), set the paths
below, then run, from the repository root:

```bash
uvx stepix run datasets/wayve_scenes/README.md
```

`debug = true` processes only the first scene, as a smoke test.

```stepix
[params]
archives = "/path/to/wayve_scenes"          # holds scene_*.zip; "" if already extracted
scenes_path = "/path/to/wayve_scenes_extracted"
output_path = "/path/to/standard_e2e_data"
split = "all"                                # passthrough label: all, train, val or test
config = "../../configs/base.yaml"
num_workers = 8
debug = false
```

### 1. Extract the archives

[`extract.sh`](extract.sh) unzips every `scene_*.zip` into `scenes_path`; files that
already exist are skipped. Set `archives = ""` if `scenes_path` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.archives} {params.scenes_path}"
cache = true
```

### 2. Check the input

Each extracted scene holds its COLMAP model in `colmap_sparse/rig/`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.scenes_path} -mindepth 4 -maxdepth 4 -path '*/colmap_sparse/rig/images.bin' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No <scene>/colmap_sparse/rig/images.bin in" {params.scenes_path} >&2
    exit 1
fi
echo "Found $n scene(s)."
'''
requires = ["extract"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/wayve_scenes/<split>/` (one `.npz` per frame plus
`index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset wayve_scenes \
    --input_path={params.scenes_path} \
    --output_path={params.output_path} \
    --split={params.split} \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
