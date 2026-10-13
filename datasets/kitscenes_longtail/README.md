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

It ships no lidar, HD map or 3D boxes.

## Splits

Each split is a set of parquet shards `data/<split>-*.parquet`.

| Split | What it holds | Default |
|---|---|---|
| `train` | 50.6 GB; the expert and counterfactual futures | on |
| `val` | 8.5 GB; no real futures | on |
| `test` | 34.0 GB; futures withheld (eval ground truth, a `[[-100, -100]]` sentinel) | on |
| `train_raw` | 43.1 GB; `train` at the native resolution | off |
| `test_raw` | 28.1 GB; `test` at the native resolution | off |

The repository also holds `val_raw`, `train_future`, `val_future`, `train_few_shot`
and `train_raw_few_shot`, which the processor does not accept.

## Known issues

- **The intrinsics do not fit the `_raw` frames** (source data). The fixed-rig `K` from
  the dataset README, which StandardE2E attaches to every frame, matches the processed
  frames, not the native-resolution `_raw` ones.

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

`debug = true` processes only the first scenario of each split, as a smoke test.
`download = true` fetches the enabled splits from Hugging Face first (step 1).

```stepix
[params]
input_path = "/path/to/KITScenes-LongTail"   # the dataset root (holds data/) or data/ itself
output_path = "/path/to/standard_e2e_data"
config = "../../configs/kitscenes_longtail.yaml"
num_workers = 8
download = false                             # true: fetch the data first (step 1)
debug = false
# Splits to preprocess, one step each (see Splits):
train = true
val = true
test = true
train_raw = false
test_raw = false
```

### 1. Download (optional)

With `download = true`, this step fetches the enabled splits' parquet shards from the
gated [Hugging Face repo](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail)
into `input_path` with the `hf` CLI (run through `uvx`). Accept the dataset's terms on
Hugging Face and log in first (`hf auth login`, or set `HF_TOKEN`); otherwise the
download is refused. An interrupted download resumes.

```stepix
[steps.download]
env = "."
run = '''
if [ {params.download} != true ]; then
    echo "download = false: nothing to download."
    exit 0
fi
splits=""
if [ {params.train} = true ]; then splits="$splits train"; fi
if [ {params.val} = true ]; then splits="$splits val"; fi
if [ {params.test} = true ]; then splits="$splits test"; fi
if [ {params.train_raw} = true ]; then splits="$splits train_raw"; fi
if [ {params.test_raw} = true ]; then splits="$splits test_raw"; fi
set --
for split in $splits; do
    set -- "$@" --include "data/$split-*"
done
if [ $# -eq 0 ]; then
    echo "No split is set to true: nothing to download."
    exit 0
fi
uvx --from 'huggingface_hub>=0.34' hf download KIT-MRT/KITScenes-LongTail --repo-type dataset \
    "$@" --local-dir {params.input_path} || {
    echo "Download failed. If access was denied, accept the terms at" >&2
    echo "https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail and log in with 'hf auth login' (or set HF_TOKEN)." >&2
    exit 1
}
'''
cache = true
```

### 2. Check the input

A split is the `<split>-*.parquet` shards in `data/` (or directly in `input_path`).

```stepix
[steps.check]
env = "."
run = '''
splits=""
if [ {params.train} = true ]; then splits="$splits train"; fi
if [ {params.val} = true ]; then splits="$splits val"; fi
if [ {params.test} = true ]; then splits="$splits test"; fi
if [ {params.train_raw} = true ]; then splits="$splits train_raw"; fi
if [ {params.test_raw} = true ]; then splits="$splits test_raw"; fi
if [ -z "$splits" ]; then
    echo "No split is set to true." >&2
    exit 1
fi
data={params.input_path}/data
[ -d "$data" ] || data={params.input_path}
for split in $splits; do
    n=$(find "$data" -maxdepth 1 -name "$split-*.parquet" | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "No $split-*.parquet shards in" "$data" >&2
        exit 1
    fi
    echo "Found $n $split shard(s)."
done
'''
requires = ["download"]
cache = true
```

### 3. Preprocess

One step per split: each writes `<output_path>/kitscenes_longtail/<split>/` (one `.npz` per frame
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path={params.input_path} \
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path={params.input_path} \
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
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=test \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_train_raw]
env = "."
run = '''
if [ {params.train_raw} != true ]; then
    echo "train_raw = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=train_raw \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```

```stepix
[steps.preprocess_test_raw]
env = "."
run = '''
if [ {params.test_raw} != true ]; then
    echo "test_raw = false: skipped."
    exit 0
fi
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_longtail \
    --input_path={params.input_path} \
    --output_path={params.output_path} \
    --split=test_raw \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
cache = true
```
