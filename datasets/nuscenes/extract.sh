#!/usr/bin/env bash
# Extract the nuScenes archives into a single dataroot the processor reads.
#
# nuScenes ships as .tgz files that each expand to maps/ samples/ sweeps/
# v1.0-*/ at the archive root, so extracting them all into the SAME dataroot
# reconstructs the official layout:
#   v1.0-mini.tgz                                   -> v1.0-mini/  (meta + blobs)
#   v1.0-trainval_meta.tgz + v1.0-trainval*_blobs.tgz -> v1.0-trainval/ + samples/
#   v1.0-test_meta.tgz     + v1.0-test_blobs.tgz      -> v1.0-test/     + samples/
# The separate map-expansion pack (nuScenes-map-expansion-v*.zip), when present,
# is unzipped into <dataroot>/maps/; without it the HD map is skipped.
#
# The extracted copy is ~the same size as the archives (jpg/bin are already
# compressed), so budget roughly as much free space as the download and write
# it somewhere OTHER than the read-only dataset mount.
#
# Usage:
#   extract.sh <ARCHIVE_SRC> <DATAROOT>
#     ARCHIVE_SRC  a single .tgz, or a directory holding the nuScenes archives;
#                  "" when DATAROOT is already extracted (nothing to do)
#     DATAROOT     output dir (will hold <version>/, samples/, sweeps/, maps/)
set -euo pipefail

ARCHIVE_SRC="${1?usage: extract.sh <ARCHIVE_SRC> <DATAROOT>}"
DATAROOT="${2:?usage: extract.sh <ARCHIVE_SRC> <DATAROOT>}"
if [[ -z "$ARCHIVE_SRC" ]]; then
    echo "No archives given; using the extracted dataroot $DATAROOT as is."
    exit 0
fi
mkdir -p "$DATAROOT"

shopt -s nullglob
if [[ -f "$ARCHIVE_SRC" ]]; then
    archives=("$ARCHIVE_SRC")
    expansions=()
else
    archives=("$ARCHIVE_SRC"/*.tgz "$ARCHIVE_SRC"/*.tar.gz "$ARCHIVE_SRC"/*.tar)
    expansions=("$ARCHIVE_SRC"/nuScenes-map-expansion-v*.zip)
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
for expansion in "${expansions[@]}"; do
    echo "extracting $(basename "$expansion") into maps/ ..."
    mkdir -p "$DATAROOT/maps"
    unzip -n -q "$expansion" -d "$DATAROOT/maps"
done

echo "Done. Extracted to $DATAROOT"
