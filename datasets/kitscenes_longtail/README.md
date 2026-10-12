# KITScenes LongTail

[KITScenes LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail)
(KIT / MRT, arXiv:2603.23607) is the **long-tail / reasoning-traces** sibling of
[KITScenes Multimodal](../kitscenes_multimodal/README.md): ~1000 9 s scenarios filtered
for **rare events** (adverse weather, construction, night, road closures, overtaking and
lane changes), each with expert and counterfactual futures and multilingual reasoning
traces. It ships as Hugging Face `datasets` **parquet**, one scenario per row.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 6 surround ring cameras | — | — | — | ✓ (driving instruction) | ✓ (counterfactual futures) |

## What StandardE2E extracts

Each scenario is a 360° multi-view video, so it is **unrolled into one frame per
timestep**: one **frame = one timestep** (5 Hz), one **segment = one scenario** (~21
frames over the 4 s observation window).

- **cameras**: the six ring cameras at that timestep, mapped to the canonical
  `CameraDirection` surround members with the fixed-rig pinhole `K` and
  `T_ego_from_camera` from the dataset README's `(K, R, t)` (where `R` is ego→camera,
  so the extrinsics are `inv(se3(R, t))`).
- **intent**: the high-level `driving_instruction` folded into the coarse `Intent`
  (`left` → `GO_LEFT`, `right` → `GO_RIGHT`, `straight` → `GO_STRAIGHT`, else
  `UNKNOWN`); the raw instruction is kept in `aux_data`.
- **past_states**: the ego history up to that timestep, re-expressed ego-relative to
  the frame (ego at the origin, facing +x along its path tangent), in metres (x forward,
  y left).
- **future_states / preference_trajectory**: the prediction targets, the 5 s expert
  future (25 points) and the counterfactual futures that are present (`wrong_speed`,
  `neglect_instruction`, `off_road`, `crash`), are attached to the **`t = 0` (last)
  frame** only, marked `is_prediction_frame` in the index.
- **aux_data**: the English reasoning trace (situational awareness plus per-phase
  acceleration and steering rationale).

It ships no lidar, HD map or 3D boxes. The released `test` split withholds the futures
(eval ground truth, encoded as a `[[-100, -100]]` sentinel); only `train` and
`train_raw` carry real expert and counterfactual futures. The `_raw` splits hold the
native-resolution frames; the camera intrinsics match the non-raw (processed) frames.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable), without its training-only dependencies (PyTorch), which
preprocessing never imports. LongTail has no devkit: the parquet rows are streamed
with pyarrow, and the camera parameters published in the dataset README are vendored.
[`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/kitscenes_longtail <command>`.

## Pipeline

Download a split from the gated Hugging Face repo (accept the terms first), e.g.:

```bash
hf download KIT-MRT/KITScenes-LongTail --repo-type dataset --include "data/train-*" --local-dir /path/to/KITScenes-LongTail
```

then set the paths below and run, from the repository root:

```bash
uvx stepix run datasets/kitscenes_longtail/README.md
```

`debug = true` processes only the first scenario of the split, as a smoke test.

```stepix
[params]
input_path = "/path/to/KITScenes-LongTail"   # the dataset root (holds data/) or data/ itself
output_path = "/path/to/standard_e2e_data"
split = "train"                               # train, test, train_raw, test_raw or val
config = "../../configs/kitscenes_longtail.yaml"
num_workers = 8
debug = false
```

### 1. Check the input

A split is the `<split>-*.parquet` shards in `data/` (or directly in `input_path`).

```stepix
[steps.check]
env = "."
run = '''
data={params.input_path}/data
[ -d "$data" ] || data={params.input_path}
n=$(find "$data" -maxdepth 1 -name {params.split}'-*.parquet' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No {params.split}-*.parquet shards in" "$data" >&2
    exit 1
fi
echo "Found $n {params.split} shard(s)."
'''
cache = true
```

### 2. Preprocess

Writes `<output_path>/kitscenes_longtail/<split>/` (one `.npz` per frame plus
`index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
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
