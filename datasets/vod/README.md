# View-of-Delft

[View-of-Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/)
(TU Delft, IEEE RA-L 2022) is a compact **urban** dataset whose distinctive sensor is a
**3+1D radar**, recorded alongside a front camera, a 64-layer Velodyne lidar and
KITTI-format 3D labels.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 1 front (1936×1216 pinhole) | Velodyne HDL-64 | — | ✓ | — | — |

## What StandardE2E extracts

- **One frame = one keyframe**; **one segment = one recording scene** (`delft_*`),
  grouped via the official scene table so the per-segment past/future ego trajectory
  never spans two recordings.
- **cameras**: the single front camera (`CameraDirection.FRONT`; intrinsics from the
  calibration's `P2`, extrinsics from `inv(Tr_velo_to_cam)`).
- **lidar_pc**: the Velodyne cloud (xyz) in the ego (`velodyne`) frame; per-point
  reflectance is dropped.
- **detections_3d**: the KITTI labels moved from camera coordinates into the ego frame,
  each of VoD's 13 classes folded into the coarse `DetectionType` (the two-wheeler
  family `bicycle` / `Cyclist` / `rider` / `moped_scooter` / `motor` → `BICYCLE`;
  `Car` / `truck` / `vehicle_other` → `VEHICLE`; static or ambiguous boxes → `UNKNOWN`;
  `DontCare` dropped). Box yaw is VoD's KITTI rotation about the lidar −Z axis with a
  camera-x zero reference, so the ego heading is `−(rotation + π/2)`.
- **past/future ego trajectories**: from the per-frame `mapToCamera` pose, via the
  `FuturePastStatesFromMatricesAggregator`.

The 3+1D radar (`radar` / `radar_3frames` / `radar_5frames`) has no StandardE2E modality
yet and is not ingested. The detection release ships no per-frame timestamps, so they
are synthesised at the 10 Hz lidar rate.

## Splits

The detection release ships three splits as official scene assignments; the pipeline
groups frames into scenes with the scene table from the devkit's documentation.

| Split | What it holds | Default |
|---|---|---|
| `train` | 13 labelled recording scenes (`lidar/training/`) | on |
| `val` | 4 labelled scenes (`lidar/training/`) | on |
| `test` | 7 scenes (`lidar/testing/`); labels withheld, so the frames carry no boxes | on |

## Known issues

- **Three frames have no map pose** (source data). `pose/00544.json` and
  `pose/04652.json` hold only `odomToCamera`, and `pose/01312.json` is empty, so these
  frames, the first of `delft_2`, `delft_3` and `delft_14`, cannot be placed on their
  scene's trajectory. StandardE2E skips them with a warning (the official devkit's pose
  reader fails on them).

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable), without its training-only dependencies (PyTorch), which
preprocessing never imports. The frame math (calibration, poses, label parsing) is
implemented in StandardE2E with NumPy, following the
[VoD devkit](https://github.com/tudelft-iv/view-of-delft-dataset)'s conventions.
[`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/vod <command>`.

## Pipeline

Request and download `view_of_delft_detection_PUBLIC.zip` (and, optionally,
`label_2_with_track_ids.zip`) from the
[VoD website](https://intelligent-vehicles.org/datasets/view-of-delft/), set the paths
below, then run, from the repository root:

```bash
uvx stepix run datasets/vod/README.md
```

`debug = true` processes only the first scene of each split, as a smoke test.

```stepix
[params]
detection_zip = "/path/to/view_of_delft_detection_PUBLIC.zip"   # "" if already extracted
track_ids_zip = "/path/to/label_2_with_track_ids.zip"           # "" to keep the base labels
extracted_path = "/path/to/vod_extracted"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/vod.yaml"
num_workers = 8
debug = false
# Splits to preprocess, one step each (see Splits):
train = true
val = true
test = true
```

### 1. Extract the archives

[`extract.sh`](extract.sh) unpacks only the `lidar/` tree of the detection release (the
radar trees are skipped, which turns a ~14 GB unzip into a far smaller one) and, when
`track_ids_zip` is set, overlays the labels that carry per-object track ids. Existing
files are kept. Set `detection_zip = ""` if `extracted_path` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.detection_zip} {params.extracted_path} {params.track_ids_zip}"
cache = true
```

### 2. Check the input

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
root={params.extracted_path}/view_of_delft_PUBLIC/lidar
for split in $splits; do
    case $split in
        test) subdir=testing ;;
        *) subdir=training ;;
    esac
    if [ ! -d "$root/$subdir/velodyne" ]; then
        echo "No lidar/$subdir/velodyne/ (needed by $split) in" "$root" >&2
        exit 1
    fi
    echo "Found lidar/$subdir/ for $split."
done
'''
requires = ["extract"]
cache = true
```

### 3. Preprocess

One step per split: each writes `<output_path>/vod/<split>/` (one `.npz` per frame
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset vod \
    --input_path={params.extracted_path}/view_of_delft_PUBLIC \
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset vod \
    --input_path={params.extracted_path}/view_of_delft_PUBLIC \
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset vod \
    --input_path={params.extracted_path}/view_of_delft_PUBLIC \
    --output_path={params.output_path} \
    --split=test \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```
