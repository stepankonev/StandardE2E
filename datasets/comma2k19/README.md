# comma2k19

[comma2k19](https://github.com/commaai/comma2k19) is a large highway-driving
dataset from comma.ai: ~2000 one-minute segments (~33 hours), each a single
forward-facing 20 Hz road camera (the comma EON, 1164×874) with a fused GNSS/IMU
ego pose and CAN telemetry.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 1 front (1164×874 pinhole) | — | — | — | — | — |

## What StandardE2E extracts

- **One frame = one video frame** of a segment; **one segment = one ~1-minute
  segment** (`<dongle>|<route>/<segment>`).
- **cameras**: the single `CameraDirection.FRONT` image, with the EON pinhole
  intrinsics and identity extrinsics (the dataset's pose *is* the camera pose).
- **global_position**: the ego pose derived from the ECEF `global_pose` into a
  per-segment local FLU frame (x-forward / y-left / z-up, origin at the segment
  start), so absolute X/Y/Z and heading are segment-relative. It also carries the
  ego **speed** (`TrajectoryComponent.SPEED`) from the ECEF velocity.
- **past/future ego trajectories**: from the per-frame poses, via the
  `FuturePastStatesFromMatricesAggregator`; being ego-relative, they are
  unaffected by the segment-local origin.

It ships no lidar, HD map, 3D boxes or driving command. Each `video.hevc` is
decoded frame by frame, forward-only, because HEVC random seek is unreliable.

**Output volume.** 20 Hz × ~2000 segments ≈ 2.4 M frames (~2 TB at the native
resolution). Two knobs bound it: the `frame_stride` param below keeps every N-th
frame (`1` = full 20 Hz, `4` ≈ 5 Hz), and the `cameras_identity_adapter`'s
`max_size` param in [`configs/comma2k19.yaml`](../../configs/comma2k19.yaml)
downscales each frame so its longest side is at most that many pixels
(intrinsics scaled to match).

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project:
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable), without its training-only dependencies (PyTorch), which
preprocessing never imports. comma2k19 needs no SDK: OpenCV decodes the video.
[`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/comma2k19 <command>`.

## Pipeline

Download the ten `Chunk_*.zip` archives (see the
[comma2k19 repository](https://github.com/commaai/comma2k19)), set the paths
below, then run, from the repository root:

```bash
uvx stepix run datasets/comma2k19/README.md
```

`debug = true` processes only the first segment, as a smoke test.

```stepix
[params]
zip_dir = "/path/to/comma2k19"                # holds Chunk_*.zip; "" if already extracted
extracted_path = "/path/to/comma2k19_extracted"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/comma2k19.yaml"
frame_stride = 1
num_workers = 8
debug = false
```

### 1. Extract the archives

[`extract.sh`](extract.sh) unzips every chunk into `extracted_path`; files that
already exist are skipped. Extract only the chunks you intend to process, and set
`zip_dir = ""` if `extracted_path` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.zip_dir} {params.extracted_path}"
cache = true
```

### 2. Check the input

Each extracted segment holds its pose files in `global_pose/` next to `video.hevc`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.extracted_path} -path '*/global_pose/frame_positions' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No <segment>/global_pose/frame_positions in" {params.extracted_path} >&2
    exit 1
fi
echo "Found $n segment(s)."
'''
requires = ["extract"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/comma2k19/all/` (one `.npz` per frame plus
`index.parquet`). comma2k19 has no official split, so the split is the
passthrough label `all`.

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset comma2k19 \
    --input_path={params.extracted_path} \
    --output_path={params.output_path} \
    --split=all \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --frame_stride={params.frame_stride} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
