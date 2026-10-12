#!/usr/bin/env bash
# Extract the WayveScenes101 scene_*.zip archives into one scenes directory.
#
# Each archive unpacks to a scene_<NNN>/ folder (colmap_sparse/rig/, images/,
# masks/), which is the layout the processor reads. The extracted copy is about
# as large as the zips, so write it somewhere other than a read-only dataset
# mount. Re-runs are idempotent: unzip -n skips files that already exist.
#
# Usage: extract.sh <ZIP_DIR> <SCENES_PATH>
#   ZIP_DIR      directory holding the downloaded scene_*.zip files;
#                "" when SCENES_PATH is already extracted (nothing to do)
#   SCENES_PATH  output directory
set -euo pipefail

ZIP_DIR="${1?usage: extract.sh <ZIP_DIR> <SCENES_PATH>}"
SCENES_PATH="${2:?usage: extract.sh <ZIP_DIR> <SCENES_PATH>}"
if [[ -z "$ZIP_DIR" ]]; then
    echo "No archives given; using the extracted scenes in $SCENES_PATH as is."
    exit 0
fi

shopt -s nullglob
zips=("$ZIP_DIR"/scene_*.zip)
if [[ ${#zips[@]} -eq 0 ]]; then
    echo "No scene_*.zip found in $ZIP_DIR" >&2
    exit 1
fi

mkdir -p "$SCENES_PATH"
for zip in "${zips[@]}"; do
    echo "extracting $(basename "$zip") ..."
    unzip -n -q "$zip" -d "$SCENES_PATH"
done
echo "Done. Extracted ${#zips[@]} scene(s) into $SCENES_PATH"
