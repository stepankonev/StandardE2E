# NATIX Multi-Camera

[NATIX Multi-Camera](https://huggingface.co/datasets/natix-network-org/natix-multi-camera-driving-dataset)
is a **crowd-sourced** driving dataset: 100 h of **Tesla-dashcam** surround footage
from everyday drivers in Switzerland and the United States, as ~1-minute mp4 clips at
~36 fps with per-clip **consumer-grade GPS** (1-10 Hz), per-trip camera calibration and
trip metadata. Its curated edge-case sibling has its own folder,
[NATIX Edge Case](../natix_edgecase/README.md).

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 4- or 6-camera Tesla rig | — | — | — | — | — |

## What StandardE2E extracts

- **One frame = one front-camera GPS fix** (the video between fixes has no pose and is
  not emitted; `frame_stride` subsamples further). **One segment = one trip piece**
  (`<trip-id>_<n>[_seqNN]`, continuous minutes), its clips concatenated chronologically,
  with the wall-clock CSV timestamps localized via `trip_insight.json`'s IANA timezone.
- **cameras**: mapped to the canonical `CameraDirection` by **facing**, verified from the
  calibration rotations: in 4-camera trips `LEFT` / `RIGHT` *are* the backward-facing
  repeaters → `REAR_LEFT` / `REAR_RIGHT`, and the 6-camera pillar pair faces forward-side
  → `FRONT_LEFT` / `FRONT_RIGHT`. Pinhole `K` and Brown-Conrady distortion as shipped
  (the front camera can be natively 2896×1876; its `K` already matches); extrinsics
  `T_ego_from_camera` in the optical frame, converted from the body-FLU rotations and
  centimetres. Streams differ in frame rate and duration, so each camera is matched to
  the fix **by timestamp**; a camera with no usable match (>1 s off) is absent from that
  frame.
- **global_position**: GPS-derived, through a per-segment azimuthal-equidistant local
  east/north frame anchored at the segment's first fix (z = 0), with yaw from
  `heading_deg`; unreliable headings (missing, or at ≤0.5 m/s) are held from the last
  reliable fix. It also carries the fix **speed**.
- **past/future ego trajectories**: from the GPS poses, via the
  `FuturePastStatesFromMatricesAggregator`.
- **extra_index_data**: trip, country, region and camera count, for filtering.

No lidar, HD map, 3D boxes or driving command, and no canonical split (`all` is a
passthrough label). Source-data quality bounds, documented and not corrected: GPS
accuracy is tens of metres, video/GPS sync error 0-1 s (extremes to 3 s), and
calibrations are nominal per vehicle model.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and pyproj, which builds the local GPS frame, without StandardE2E's
training-only dependencies (PyTorch), which preprocessing never imports. NATIX has no
devkit. [`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/natix_multicam <command>`.

## Pipeline

The gated Hugging Face repo (accept the terms first) carries a ~20-minute
`dataset-sample/`:

```bash
hf download natix-network-org/natix-multi-camera-driving-dataset --repo-type dataset --local-dir /path/to/natix-multi-camera-driving-dataset
```

The full ~1.28 TB release is pulled from a Cloudflare R2 bucket with the credentials
granted on access approval (see the dataset card). Nothing needs extracting. Set the
paths below, then run, from the repository root:

```bash
uvx stepix run datasets/natix_multicam/README.md
```

`debug = true` processes only the first trip, as a smoke test.

```stepix
[params]
input_path = "/path/to/natix-multi-camera-driving-dataset"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/natix_multicam.yaml"
frame_stride = 1
num_workers = 8
debug = false
```

### 1. Check the input

Trips are found by their `fixed_metadata.json`, at any depth (US trips add a state
folder, Swiss ones don't).

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.input_path} -name fixed_metadata.json | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No trips (fixed_metadata.json) under" {params.input_path} >&2
    exit 1
fi
echo "Found $n trip(s)."
'''
cache = true
```

### 2. Preprocess

Writes `<output_path>/natix_multicam/all/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset natix_multicam \
    --input_path={params.input_path} \
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
