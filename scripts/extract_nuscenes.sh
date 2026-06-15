#!/usr/bin/env bash
# Extract the nuScenes tar archives into a single dataroot the processor reads.
#
# nuScenes ships as .tgz files that each expand to maps/ samples/ sweeps/
# v1.0-*/ at the archive root, so extracting them all into the SAME dataroot
# reconstructs the official layout:
#   v1.0-mini.tgz                                   -> v1.0-mini/  (meta + blobs)
#   v1.0-trainval_meta.tgz + v1.0-trainval*_blobs.tgz -> v1.0-trainval/ + samples/
#   v1.0-test_meta.tgz     + v1.0-test_blobs.tgz      -> v1.0-test/     + samples/
#
# The extracted copy is ~the same size as the archives (jpg/bin are already
# compressed), so budget roughly as much free space as the download and write
# it somewhere OTHER than the read-only dataset mount.
#
# Usage:
#   scripts/extract_nuscenes.sh <ARCHIVE_SRC> <DATAROOT>
#     ARCHIVE_SRC  a single .tgz, or a directory holding the nuScenes .tgz files
#     DATAROOT     output dir (will hold <version>/, samples/, sweeps/, maps/)
set -euo pipefail

ARCHIVE_SRC="${1:?usage: extract_nuscenes.sh <ARCHIVE_SRC> <DATAROOT>}"
DATAROOT="${2:?usage: extract_nuscenes.sh <ARCHIVE_SRC> <DATAROOT>}"
mkdir -p "$DATAROOT"

shopt -s nullglob
if [[ -f "$ARCHIVE_SRC" ]]; then
    archives=("$ARCHIVE_SRC")
else
    archives=("$ARCHIVE_SRC"/*.tgz "$ARCHIVE_SRC"/*.tar.gz "$ARCHIVE_SRC"/*.tar)
fi
if [[ ${#archives[@]} -eq 0 ]]; then
    echo "No nuScenes archives found at $ARCHIVE_SRC" >&2
    exit 1
fi

for archive in "${archives[@]}"; do
    echo "extracting $(basename "$archive") ..."
    # --skip-old-files makes re-runs idempotent (already-extracted files kept).
    tar --skip-old-files -xf "$archive" -C "$DATAROOT"
done

echo "Done. Extracted to $DATAROOT"
