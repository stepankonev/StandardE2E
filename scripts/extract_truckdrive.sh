#!/usr/bin/env bash
# Extract the per-scene / per-modality TruckDrive zips into the devkit layout
# the processor expects. Each modality zip expands into a folder of the same
# name, reproducing the official layout:
#   <scene>/camera.zip       -> <scene>/camera/leopard/<cam>/images/...
#   <scene>/lidar.zip        -> <scene>/lidar/aeva/joint_lidars/points/...
#   <scene>/annotations.zip  -> <scene>/annotations/bounding_boxes/...
#   <scene>/calibrations.zip -> <scene>/calibrations/calib_*.json
#   <scene>/poses.zip        -> <scene>/poses/gt_trajectory.txt
#
# The extracted copy is ~the same size as the zips (jpg/bin are already
# compressed), so budget roughly as much free space as the download and write
# it somewhere OTHER than the read-only dataset mount.
#
# Usage:
#   scripts/extract_truckdrive.sh <ZIP_ROOT> <EXTRACTED_ROOT> [MODALITIES...]
#     ZIP_ROOT        dir holding scene_*/ with the *.zip files
#     EXTRACTED_ROOT  output dir (will hold scene_*/<modality>/...)
#     MODALITIES      optional subset; default: the modalities the processor
#                     reads (calibrations poses camera lidar annotations).
set -euo pipefail

ZIP_ROOT="${1:?usage: extract_truckdrive.sh <ZIP_ROOT> <EXTRACTED_ROOT> [modalities...]}"
EXTRACTED_ROOT="${2:?usage: extract_truckdrive.sh <ZIP_ROOT> <EXTRACTED_ROOT> [modalities...]}"
shift 2 || true

MODALITIES=("$@")
if [[ ${#MODALITIES[@]} -eq 0 ]]; then
    # Add ``radar`` / ``accumulated_gt_depth`` here if you need them; the
    # StandardE2E processor only reads the five below.
    MODALITIES=(calibrations poses camera lidar annotations)
fi

shopt -s nullglob
scene_dirs=("$ZIP_ROOT"/scene_*)
if [[ ${#scene_dirs[@]} -eq 0 ]]; then
    echo "No scene_* directories under $ZIP_ROOT" >&2
    exit 1
fi

for scene_dir in "${scene_dirs[@]}"; do
    scene="$(basename "$scene_dir")"
    for m in "${MODALITIES[@]}"; do
        zip="$scene_dir/$m.zip"
        if [[ ! -f "$zip" ]]; then
            echo "skip (no $m.zip): $scene"
            continue
        fi
        dest="$EXTRACTED_ROOT/$scene/$m"
        mkdir -p "$dest"
        echo "extracting $scene/$m.zip ..."
        unzip -n -q "$zip" -d "$dest"
    done
done

echo "Done. Extracted to $EXTRACTED_ROOT"
