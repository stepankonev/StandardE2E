# Datasets

Each supported dataset has a folder here with everything needed to turn its raw
download into StandardE2E's processed format:

- `README.md`: what the dataset is, what StandardE2E extracts from it, and the
  preprocessing **pipeline**, as [stepix](https://github.com/stepankonev/stepix) steps;
- `pyproject.toml`, `uv.lock` and `.python-version`: the dataset's own isolated
  [uv](https://docs.astral.sh/uv/) environment, with StandardE2E (this checkout) plus
  only the SDK that dataset needs, so no two datasets' dependencies ever meet;
- helper scripts the pipeline calls, such as `extract.sh`.

The training environment is the default one at the repository root (`uv sync`): it has
PyTorch and no dataset SDKs. Preprocessing never imports PyTorch, so the dataset
environments leave it out.

## Running a pipeline

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (Linux or
macOS), set the paths in the `[params]` block of the dataset's README, then run, from
the repository root:

```bash
uvx stepix run datasets/<name>/README.md
```

Each pipeline checks that the input looks right, extracts archives where the dataset
ships them, and preprocesses one split into `<output_path>/<name>/<split>/`. On the first
run stepix also builds the dataset's environment in `datasets/<name>/.venv/`. Steps that
succeeded are skipped on later runs unless a param they use changed; `--no-cache` reruns
them, and `--step NAME` / `--from NAME` select steps. Logs and state go to
`datasets/<name>/.stepix/`. Every README has a `debug` param that processes a single
unit (file, log or scene) as a smoke test.

| Dataset | Folder | SDK in the environment | Python |
|---|---|---|---|
| Waymo Open E2E | [`waymo_e2e`](waymo_e2e/README.md) | TensorFlow | 3.12 (≤ 3.13) |
| Waymo Open Perception | [`waymo_perception`](waymo_perception/README.md) | TensorFlow | 3.12 (≤ 3.13) |
| Argoverse 2 Sensor | [`av2_sensor`](av2_sensor/README.md) | av2 | 3.12 |
| Argoverse 2 Lidar | [`av2_lidar`](av2_lidar/README.md) | av2 | 3.12 |
| NAVSIM | [`navsim`](navsim/README.md) | nuplan-devkit | 3.12 (≤ 3.14) |
| nuScenes | [`nuscenes`](nuscenes/README.md) | nuscenes-devkit | 3.12 |
| WayveScenes101 | [`wayve_scenes`](wayve_scenes/README.md) | — | 3.12 (≤ 3.14) |
| comma2k19 | [`comma2k19`](comma2k19/README.md) | — | 3.12 (≤ 3.14) |
| TruckDrive | [`truckdrive`](truckdrive/README.md) | — | 3.12 (≤ 3.14) |
| View-of-Delft | [`vod`](vod/README.md) | — | 3.12 (≤ 3.14) |
| KITScenes Multimodal | [`kitscenes_multimodal`](kitscenes_multimodal/README.md) | pyproj | 3.12 (≤ 3.14) |
| KITScenes LongTail | [`kitscenes_longtail`](kitscenes_longtail/README.md) | — | 3.12 (≤ 3.14) |
| NATIX Multi-Camera | [`natix_multicam`](natix_multicam/README.md) | pyproj | 3.12 (≤ 3.14) |
| NATIX Edge Case | [`natix_edgecase`](natix_edgecase/README.md) | pyproj | 3.12 (≤ 3.14) |

The Python column is the version `.python-version` pins, with the newest one the
environment's `requires-python` allows in parentheses.

## Working with an environment

Any command can run in a dataset's environment, e.g. its tests:

```bash
uv run --project datasets/nuscenes python -m pytest tests/dataset_processors/test_nuscenes_dataset_processor.py
```

An environment's `uv.lock` records StandardE2E's own dependencies too, so after
changing them in the root `pyproject.toml`, re-lock every dataset:

```bash
for d in datasets/*/; do uv lock --project "$d"; done
```

## Adding a dataset

A new dataset needs its processor and converter in
`standard_e2e/caching/src_datasets/<name>/` (see the
[guide](../standard_e2e/caching/src_datasets/adding_new_dataset.md)), an entry in the
`DATASETS` registry of `standard_e2e/caching/process_source_dataset.py`, and a folder
here shaped like the others. The SDK it needs goes into its `pyproject.toml` only; the
modules that import that SDK live in its package, and its tests skip themselves
(`pytest.importorskip`) where the SDK is missing.
