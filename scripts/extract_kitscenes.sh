#!/usr/bin/env bash
# Extract KITScenes scene tarballs into a flat scenes directory.
#
# Generic untar helper shared across KITScenes variants (Multimodal today,
# LongTail later) -- the per-scene tar layout is identical. KITScenes ships on
# Hugging Face as data/<split>/<scene_uuid>.tar; each tar unpacks to a
# <scene_uuid>/ directory. This script extracts every *.tar found under SRC into
# DEST/<scene_uuid>/ (idempotent: a scene whose calibration is already present is
# skipped).
#
# Usage: extract_kitscenes.sh <SRC_dir_with_tars> <DEST_scenes_dir>
set -euo pipefail

SRC="${1:?usage: extract_kitscenes.sh <SRC_dir_with_tars> <DEST_scenes_dir>}"
DEST="${2:?usage: extract_kitscenes.sh <SRC_dir_with_tars> <DEST_scenes_dir>}"
mkdir -p "$DEST"

shopt -s nullglob globstar
tars=("$SRC"/**/*.tar)
if [ ${#tars[@]} -eq 0 ]; then
    echo "No *.tar found under $SRC (already extracted? point --input_path at the scenes)."
    exit 0
fi

for tar_path in "${tars[@]}"; do
    uuid="$(basename "$tar_path" .tar)"
    if [ -f "$DEST/$uuid/calibration/calib.json" ]; then
        echo "skip $uuid (already extracted)"
        continue
    fi
    echo "extracting $uuid"
    tar -xf "$tar_path" -C "$DEST"
done
echo "Extracted ${#tars[@]} tar(s) into $DEST"
