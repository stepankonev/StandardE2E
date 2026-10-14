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
- **cameras**: the five fisheye cameras with their COLMAP intrinsics and distortion, and
  `T_ego_from_camera` extrinsics in the OpenCV optical frame (x right, y down, z forward).
- **lidar_pc**: no sensor lidar exists, so the scene's **COLMAP SfM** point cloud is
  used instead. It is filtered as in the official SDK (inside the 1st–99th percentile
  bounding box, reprojection error ≤ 6, track length ≥ 2), converted from OpenCV to
  FLU axes, moved into each frame's ego frame and clipped to 50 m, so it flows through
  the lidar adapters. It is photogrammetric (sparse, up to scale), not a sensor
  measurement; a frame's cloud lifted by `aux_data["pose_matrix"]` reproduces the source
  SfM cloud.
- **past/future ego trajectories**: from the COLMAP poses, via the
  `FuturePastStatesFromMatricesAggregator`.

The dataset has no perception split, so `split` is a passthrough label for the output
folder.

## Known issues

- **The official SDK cannot be installed** (SDK, verified 2026-10-14 at commit
  `2fc2fa9`). `src/wayve_scenes/` has no `__init__.py`, so `setup.py`'s `find_packages()`
  finds nothing: a regular install yields an empty wheel and an editable one does not
  import either; only `PYTHONPATH=src` works. It also runs only on pycolmap ≤ 3.11
  (`Image.cam_from_world` and `compute_bounding_box` changed in 3.12), and its loader
  imports PyTorch3D, which has no Linux wheels. StandardE2E therefore reads the
  reconstructions with pycolmap directly, as the SDK does; on `scene_001` the two agree
  (camera poses to 2.5 µm, intrinsics, distortion and timestamps exactly, ego-frame
  extrinsics to 5e-7), except for the bounding-box points below.
- **The SfM cloud's bounding box depends on the pycolmap version** (COLMAP, verified).
  `compute_bounding_box(0.01, 0.99)` keeps 6 more boundary points of `scene_001` with
  pycolmap 4.2.1 (136,193 points) than with the SDK's pycolmap 3.11 (136,187).

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and [pycolmap](https://pypi.org/project/pycolmap/), COLMAP's
Python bindings, which read the reconstructions as in the official
[WayveScenes SDK](https://github.com/wayveai/wayve_scenes) (the SDK itself cannot be
installed, see [Known issues](#known-issues)), without StandardE2E's training-only
dependencies (PyTorch), which
preprocessing never imports. pycolmap publishes wheels for Linux x86-64, macOS on Apple
silicon (14+) and Windows x64 only, so the environment does not install on other
platforms. [`.python-version`](.python-version) pins Python 3.12 (`requires-python`
allows 3.12–3.14). stepix builds the environment into `.venv/` here on first use; to
run a command in it yourself:
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
