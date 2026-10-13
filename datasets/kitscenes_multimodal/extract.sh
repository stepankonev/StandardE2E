#!/usr/bin/env bash
# Extract KITScenes Multimodal scene tarballs, keeping their split folders.
#
# KITScenes Multimodal ships on Hugging Face as data/<split>/<scene_uuid>.tar;
# each tar unpacks to a <scene_uuid>/ directory. This script extracts every *.tar
# found under SRC into DEST/<split>/<scene_uuid>/, so the processor can select
# scenes by split; a tar outside a split folder goes to DEST/<scene_uuid>/.
# Re-runs are idempotent: a scene whose calibration is already present is skipped.
#
# Usage: extract.sh <SRC_dir_with_tars> <DEST_scenes_dir>
#   SRC_dir_with_tars  "" when DEST_scenes_dir is already extracted (nothing to do)
set -euo pipefail

SRC="${1?usage: extract.sh <SRC_dir_with_tars> <DEST_scenes_dir>}"
DEST="${2:?usage: extract.sh <SRC_dir_with_tars> <DEST_scenes_dir>}"
if [[ -z "$SRC" ]]; then
    echo "No archives given; using the extracted scenes in $DEST as is."
    exit 0
fi
mkdir -p "$DEST"

shopt -s nullglob globstar
tars=("$SRC"/**/*.tar)
if [ ${#tars[@]} -eq 0 ]; then
    echo "No *.tar found under $SRC (already extracted? point --input_path at the scenes)."
    exit 0
fi

for tar_path in "${tars[@]}"; do
    uuid="$(basename "$tar_path" .tar)"
    folder="$(basename "$(dirname "$tar_path")")"
    case "$folder" in
        train|val|validation|test|test_e2e|test-e2e|overlap_train_val) out="$DEST/$folder" ;;
        *) out="$DEST" ;;
    esac
    if [ -f "$out/$uuid/calibration/calib.json" ]; then
        echo "skip $uuid (already extracted)"
        continue
    fi
    echo "extracting $uuid into $out"
    mkdir -p "$out"
    tar -xf "$tar_path" -C "$out"
done
echo "Extracted ${#tars[@]} tar(s) into $DEST"
