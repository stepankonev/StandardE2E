# NATIX Edge Case

[NATIX Edge Case](https://huggingface.co/datasets/natix-network-org/natix-edge-case-driving-dataset)
is the **curated edge-case sibling** of [NATIX Multi-Camera](../natix_multicam/README.md):
rare, challenging real-world scenarios (construction zones, adverse weather, road-surface
deterioration, illegal maneuvers, obstructions) crowd-sourced from the same Tesla-dashcam
network. The first public release is 20 minutes / 86 mp4 clips across six US states
(4- and 6-camera rigs) with 21 VLM-annotated events.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 4- or 6-camera Tesla rig | — | — | — | — | — |

## What StandardE2E extracts

Footage, GPS metadata, calibration and folder layout are **identical to NATIX
Multi-Camera**, so frames, segments, camera mapping, the GPS ego pose and the data-quality
bounds are exactly as described [there](../natix_multicam/README.md) (the processor is a
subclass). What this release adds are the **edge-case annotations**
(`data/edge-case.json`): per annotated clip, one or more events, each with a `label`
(may be empty), a `[start_sec, end_sec]` window in video seconds from clip start, and a
structured VLM `ai_analysis` (event classification, visual evidence, context, agentic
validation, final detected event).

- **aux_data["edge_case_events"]**: every frame carries, inside its `.npz`, the events
  covering its timestamp: a list (empty outside events) of one plain dict per event with
  the verbatim `label`, the `start_sec` / `end_sec` window and the complete
  `ai_analysis`.
- **index**: flattened into `extra_edge_case_count` / `extra_edge_case` (summaries; a
  missing `label` falls back to the VLM `EVENT CLASSIFICATION`), so in-event frames are
  selectable with `extra_edge_case_count > 0` without opening any `.npz` (see the
  [quickstart](https://standarde2e.readthedocs.io/en/latest/quickstart.html) for an
  access snippet).

Windows map onto the frame timeline through each clip's front CSV (the epoch of video
frame 1, extrapolated when the CSV's leading rows are missing, as seen in the release).

## Known issues

Everything listed for
[NATIX Multi-Camera](../natix_multicam/README.md#known-issues) applies here too.
In addition:

- **Machine-generated annotations** (source data). The events are VLM-generated best
  effort, with integer-second windows and model-written descriptions: treat them as
  guidance, not ground truth. Some events ship an empty `label`; the index summary then
  uses the VLM's `EVENT CLASSIFICATION`.
- **A GPS freeze overlaps an event** (source data). In trip
  `4d4cfb50-69a3-45fc-aae3-8fec19120366_7` the fixes are frozen for the first ~27 s,
  across its overtake event, so the trajectories there show no motion.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and pyproj, which builds the local GPS frame, without StandardE2E's
training-only dependencies (PyTorch), which preprocessing never imports. NATIX has no
devkit. [`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/natix_edgecase <command>`.

## Pipeline

Download the release from the gated Hugging Face repo (accept the terms first; ~3.3 GB);
nothing needs extracting:

```bash
hf download natix-network-org/natix-edge-case-driving-dataset --repo-type dataset --local-dir /path/to/natix-edge-case-driving-dataset
```

then set the paths below and run, from the repository root:

```bash
uvx stepix run datasets/natix_edgecase/README.md
```

`debug = true` processes only the first trip, as a smoke test.
`download = true` fetches the release from Hugging Face first (step 1).

```stepix
[params]
input_path = "/path/to/natix-edge-case-driving-dataset"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/natix_edgecase.yaml"
frame_stride = 1
num_workers = 8
download = false  # true: fetch the data first (step 1)
debug = false
```

### 1. Download (optional)

With `download = true`, this step fetches the release (3.6 GB) from the gated
[Hugging Face repo](https://huggingface.co/datasets/natix-network-org/natix-edge-case-driving-dataset)
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
uvx --from 'huggingface_hub>=0.34' hf download natix-network-org/natix-edge-case-driving-dataset --repo-type dataset \
    --local-dir {params.input_path} || {
    echo "Download failed. If access was denied, accept the terms at" >&2
    echo "https://huggingface.co/datasets/natix-network-org/natix-edge-case-driving-dataset and log in with \'hf auth login\' (or set HF_TOKEN)." >&2
    exit 1
}
'''
cache = true
```

### 2. Check the input

Trips are found by their `fixed_metadata.json`; the events by walking up from each trip
to `edge-case.json`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.input_path} -name fixed_metadata.json | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No trips (fixed_metadata.json) under" {params.input_path} >&2
    exit 1
fi
if [ -z "$(find {params.input_path} -maxdepth 2 -name edge-case.json)" ]; then
    echo "No edge-case.json under" {params.input_path} >&2
    exit 1
fi
echo "Found $n trip(s) and the edge-case annotations."
'''
requires = ["download"]
cache = true
```

### 3. Preprocess

Writes `<output_path>/natix_edgecase/all/` (one `.npz` per frame plus `index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset natix_edgecase \
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
