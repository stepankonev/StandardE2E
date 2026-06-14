#!/usr/bin/env bash
# Extract the View-of-Delft detection release into the layout the processor reads.
#
# Only the ``lidar/`` subtree is unpacked -- camera (image_2), Velodyne cloud
# (velodyne), KITTI labels (label_2), calibration (calib), ego pose (pose) and
# the official ImageSets. The parallel ``radar`` / ``radar_3frames`` /
# ``radar_5frames`` trees are skipped: the 3+1D radar has no StandardE2E modality
# yet, and skipping them turns a ~14 GB unzip into a far smaller one.
#
# Optionally overlays the per-object track ids (``label_2_with_track_ids.zip``),
# overwriting the base labels with the identical boxes plus a track id in the
# KITTI truncation field. Write the extracted copy somewhere OTHER than the
# read-only dataset mount (budget roughly the size of the lidar tree).
#
# Usage:
#   scripts/extract_vod.sh <DETECTION_ZIP> <EXTRACTED_ROOT> [TRACK_IDS_ZIP]
#     DETECTION_ZIP   path to view_of_delft_detection_PUBLIC.zip
#     EXTRACTED_ROOT  output dir (will hold view_of_delft_PUBLIC/lidar/...)
#     TRACK_IDS_ZIP   optional label_2_with_track_ids.zip to overlay
set -euo pipefail

DETECTION_ZIP="${1:?usage: extract_vod.sh <DETECTION_ZIP> <EXTRACTED_ROOT> [TRACK_IDS_ZIP]}"
EXTRACTED_ROOT="${2:?usage: extract_vod.sh <DETECTION_ZIP> <EXTRACTED_ROOT> [TRACK_IDS_ZIP]}"
TRACK_IDS_ZIP="${3:-}"

mkdir -p "$EXTRACTED_ROOT"

echo "Extracting lidar/ tree from $DETECTION_ZIP ..."
# -n: never overwrite (idempotent re-runs). Only the lidar/ subtree.
unzip -n -q "$DETECTION_ZIP" 'view_of_delft_PUBLIC/lidar/*' -d "$EXTRACTED_ROOT"

LIDAR_ROOT="$EXTRACTED_ROOT/view_of_delft_PUBLIC/lidar"
if [[ ! -d "$LIDAR_ROOT/training/velodyne" ]]; then
    echo "Expected $LIDAR_ROOT/training/velodyne after extraction" >&2
    exit 1
fi

if [[ -n "$TRACK_IDS_ZIP" ]]; then
    echo "Overlaying track-id labels from $TRACK_IDS_ZIP ..."
    # The zip holds label_2/<frame>.txt; extracting into training/ overwrites the
    # base labels in place (-o: overwrite; identical boxes + track id).
    unzip -o -q "$TRACK_IDS_ZIP" 'label_2/*' -d "$LIDAR_ROOT/training"
fi

echo "Done. Point --input_path at: $EXTRACTED_ROOT/view_of_delft_PUBLIC"
