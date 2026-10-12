# TruckDrive

[TruckDrive](https://torc-ai.github.io/TruckDrive/) (Torc Robotics / Princeton,
CVPR 2026) is a long-range highway **heavy-truck** dataset: an 11-camera 8 MP surround
rig, seven Aeva FMCW lidars merged into a joint cloud, tracked 3D boxes and a
PPK + LiDAR-SLAM ego pose.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 11-camera 8 MP rig | Aeva FMCW joint cloud | — | ✓ | — | — |

## What StandardE2E extracts

- **One frame = one synchronization key**; **one segment = one scene**. Frames are
  matched across sensors by that key.
- **cameras**: the rig has more cameras than the eight canonical `CameraDirection`
  slots, so each camera maps to the canonical surround member matching its facing
  (`FRONT`, `FRONT_LEFT`, `SIDE_LEFT`, `REAR_LEFT`, ...) wherever one fits, with
  dedicated members only for the extra views the eight can't name: the forward
  telephoto pair `FRONT_LEFT_NARROW` / `FRONT_RIGHT_NARROW` and the rear-facing side
  pair `SIDE_LEFT_BACK` / `SIDE_RIGHT_BACK`.
- **lidar_pc**: the seven-sensor Aeva joint cloud (xyz), in the ego (`velodyne`) frame.
- **detections_3d**: the tracked boxes in the ego frame, with the ego vehicle's own
  cab and trailer and the `DontCare` groups excluded, per the paper's taxonomy.
- **past/future ego trajectories**: from the ego pose, via the
  `FuturePastStatesFromMatricesAggregator`.

The short-range Ouster lidar, the 4D radar, the accumulated ground-truth depth and the
lane lines have no StandardE2E target yet and are not ingested. 8 MP frames are large:
the `cameras_identity_adapter`'s `max_size` in
[`configs/truckdrive.yaml`](../../configs/truckdrive.yaml) downscales them.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable), without its training-only dependencies (PyTorch), which
preprocessing never imports. The
[TruckDrive devkit](https://github.com/torc-ai/TruckDrive) is not a Python package, so
its calibration-graph and file-matching logic is reproduced in StandardE2E.
[`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/truckdrive <command>`.

## Pipeline

Download the dataset with the devkit's `download_truckdrive.sh` (see its
[README](https://github.com/torc-ai/TruckDrive#downloading-the-truckdrive-dataset)), set
the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/truckdrive/README.md
```

`debug = true` processes only the first scene, as a smoke test.

```stepix
[params]
zip_root = "/path/to/TruckDrive"             # holds scene_*/ with the *.zip files; "" if extracted
extracted_path = "/path/to/truckdrive_extracted"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/truckdrive.yaml"
num_workers = 8
debug = false
```

### 1. Extract the archives

[`extract.sh`](extract.sh) unzips each scene's `calibrations`, `poses`, `camera`, `lidar`
and `annotations` zips into `extracted_path`, reproducing the devkit layout; files that
already exist are skipped. Set `zip_root = ""` if `extracted_path` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.zip_root} {params.extracted_path}"
cache = true
```

### 2. Check the input

Each extracted scene holds its ego trajectory in `poses/gt_trajectory.txt`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.extracted_path} -path '*/poses/gt_trajectory.txt' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No <scene>/poses/gt_trajectory.txt in" {params.extracted_path} >&2
    exit 1
fi
echo "Found $n scene(s)."
'''
requires = ["extract"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/truckdrive/all/` (one `.npz` per frame plus `index.parquet`).
TruckDrive has no official split, so the split is the passthrough label `all`.

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset truckdrive \
    --input_path={params.extracted_path} \
    --output_path={params.output_path} \
    --split=all \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
