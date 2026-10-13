#!/usr/bin/env bash
# Build a writable mirror of the nuPlan maps folder for NAVSIM preprocessing.
#
# nuplan-devkit's GPKGMapsDB creates a .maplocks/ directory inside the maps root,
# which fails when the dataset sits on a read-only mount. The mirror holds a copy
# of the JSON manifest and one symlink per city directory, so the map files
# themselves are not copied. Re-runs refresh the manifest and the links.
#
# Usage: maps_mirror.sh <MAPS_DIR> <MIRROR_DIR>
#   MAPS_DIR    the dataset's maps/ folder (holds nuplan-maps-v1.0.json)
#   MIRROR_DIR  a writable folder to build the mirror in
set -euo pipefail

MAPS_DIR="${1:?usage: maps_mirror.sh <MAPS_DIR> <MIRROR_DIR>}"
MIRROR_DIR="${2:?usage: maps_mirror.sh <MAPS_DIR> <MIRROR_DIR>}"
MANIFEST="nuplan-maps-v1.0.json"

if [[ ! -f "$MAPS_DIR/$MANIFEST" ]]; then
    echo "No $MANIFEST in $MAPS_DIR" >&2
    exit 1
fi
mkdir -p "$MIRROR_DIR"
if [[ "$(cd "$MAPS_DIR" && pwd -P)" == "$(cd "$MIRROR_DIR" && pwd -P)" ]]; then
    echo "The mirror must be a different folder than $MAPS_DIR" >&2
    exit 1
fi

cp "$MAPS_DIR/$MANIFEST" "$MIRROR_DIR/"
for city in "$MAPS_DIR"/*/; do
    ln -sfn "$(cd "$city" && pwd -P)" "$MIRROR_DIR/$(basename "$city")"
done
echo "Maps mirror ready at $MIRROR_DIR"
