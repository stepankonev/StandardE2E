# KITScenes Multimodal

[KITScenes Multimodal](https://www.kitscenes.com/multimodal/) (KIT / MRT,
arXiv:2606.02956) is a large **European urban** dataset (~1000 scenes at 10 Hz) whose
headline annotation is a dense, georeferenced **Lanelet2 HD map**. Its long-tail
sibling, [KITScenes LongTail](../kitscenes_longtail/README.md), has a different format
and its own folder.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 6 surround ring cameras | lidar_top | ✓ (Lanelet2) | — | — | — |

## What StandardE2E extracts

- **One frame = one synchronized 10 Hz snapshot**; **one segment = one scene** (a UUID
  directory). The reference timeline gives per-frame timestamps and `poses.txt` the
  per-frame ego pose.
- **cameras**: the six `camera_ring_*` cameras, mapped to the canonical
  `CameraDirection` surround members (pinhole `K`; `T_ego_from_camera` from the
  calibration's `T_to_reference`).
- **lidar_pc**: the 128-beam `lidar_top` cloud (xyz), de-discretized from its int32
  storage with the invalid-return sentinels dropped, in the ego `base_frame` (whose
  origin is `lidar_top`).
- **hd_map**: each scene's Lanelet2 OSM (`maps/map.osm`), parsed and translated to the
  unified `MapElementType` taxonomy (lanelet road / emergency / bicycle → `LANE_CENTER`
  from the left/right bound midpoints, `crosswalk` → `CROSSWALK`, `line_thin` /
  `line_thick` / `bike_marking` → `LANE_BOUNDARY`, `curbstone` / `road_border` →
  `ROAD_EDGE`, `stop_line` → `STOP_LINE`, `traffic_light*` → `TRAFFIC_LIGHT`), then
  cropped to an ego-centric region per frame and rasterised by `HDMapBEVAdapter`.
  `poses.txt` is already in the Lanelet2 map-local frame (UTM zone 32N minus the
  `origin.json` anchor), so no GNSS reconciliation is needed.
- **past/future ego trajectories**: from the poses, via the
  `FuturePastStatesFromMatricesAggregator`.

KITScenes ships **no 3D boxes** (the HD map is its annotation product). Not ingested:
the three "base" cameras (the hi-res front-center camera and the rectified stereo pair),
the other six lidars, the three imaging radars and the GNSS/INS streams.

Splits follow the official geo-disjoint folders (`train`, `val`, `test`, `test_e2e`,
`overlap_train_val`). With a flat layout (one folder of scenes, like the single-scene
sample, or what step 2 produces) every scene is processed and `split` is only the output
label.

## Known issues

None known.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project.
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) and pyproj, which projects the map, without StandardE2E's
training-only dependencies (PyTorch), which preprocessing never imports. Calibration,
poses, lidar and the Lanelet2 map are read by StandardE2E itself; the
[KITScenes devkit](https://github.com/KIT-MRT/kitscenes) is not on PyPI.
[`.python-version`](.python-version) pins Python 3.12 (`requires-python` allows
3.12–3.14). stepix builds the environment into `.venv/` here on first use; to run a
command in it yourself: `uv run --project datasets/kitscenes_multimodal <command>`.

## Pipeline

Download one split from the gated Hugging Face repo (accept the terms first), e.g.:

```bash
hf download KIT-MRT/KITScenes-Multimodal --repo-type dataset --include "data/val/*" --local-dir /path/to/kitscenes_multimodal_dl
```

(or `KIT-MRT/KITScenes-Multimodal-Sample` for a single scene), set the paths below, then
run, from the repository root:

```bash
uvx stepix run datasets/kitscenes_multimodal/README.md
```

`debug = true` processes only the first scene of the split, as a smoke test.
`download = true` fetches the split from Hugging Face first (step 1).

```stepix
[params]
archives = "/path/to/kitscenes_multimodal_dl"     # holds the scene tars; "" if extracted
scenes_path = "/path/to/kitscenes_multimodal_scenes"
output_path = "/path/to/standard_e2e_data"
split = "val"                    # train, val, test, test_e2e, overlap_train_val or all
config = "../../configs/kitscenes_multimodal.yaml"
num_workers = 8
download = false                                  # true: fetch the data first (step 1)
debug = false
```

### 1. Download (optional)

With `download = true`, this step fetches the `split`'s scene tars (`data/<split>/`,
e.g. 117 scenes / 508 GB for `val`; everything for `split = "all"`) from the gated
[Hugging Face repo](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) into
`archives` with the `hf` CLI (run through `uvx`); step 2 then extracts them. Accept the
dataset's terms on Hugging Face and log in first (`hf auth login`, or set `HF_TOKEN`);
otherwise the download is refused. An interrupted download resumes.

```stepix
[steps.download]
env = "."
run = '''
if [ {params.download} != true ]; then
    echo "download = false: nothing to download."
    exit 0
fi
if [ -z {params.archives} ]; then
    echo "download = true needs archives: the folder to download into." >&2
    exit 1
fi
case {params.split} in
    all) pattern='data/*' ;;
    *) pattern=data/{params.split}/'*' ;;
esac
uvx --from 'huggingface_hub>=0.34' hf download KIT-MRT/KITScenes-Multimodal --repo-type dataset \
    --include "$pattern" --local-dir {params.archives} || {
    echo "Download failed. If access was denied, accept the terms at" >&2
    echo "https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal and log in with \'hf auth login\' (or set HF_TOKEN)." >&2
    exit 1
}
'''
cache = true
```

### 2. Extract the archives

[`extract.sh`](extract.sh) extracts every scene tar found under `archives` into
`scenes_path/<scene_uuid>/`, skipping scenes already extracted. Set `archives = ""` if
`scenes_path` is already extracted.

```stepix
[steps.extract]
env = "."
run = "bash extract.sh {params.archives} {params.scenes_path}"
requires = ["download"]
cache = true
```

### 3. Check the input

Each scene folder holds its calibration in `calibration/calib.json`.

```stepix
[steps.check]
env = "."
run = '''
n=$(find {params.scenes_path} -maxdepth 5 -path '*/calibration/calib.json' | wc -l)
if [ "$n" -eq 0 ]; then
    echo "No <scene>/calibration/calib.json in" {params.scenes_path} >&2
    exit 1
fi
echo "Found $n scene(s)."
'''
requires = ["extract"]
cache = true
```

### 4. Preprocess

Writes `<output_path>/kitscenes_multimodal/<split>/` (one `.npz` per frame plus
`index.parquet`).

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset kitscenes_multimodal \
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
