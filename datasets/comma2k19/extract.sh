#!/usr/bin/env bash
# Extract the comma2k19 Chunk_*.zip archives into one directory the processor
# reads (it discovers the <dongle>|<route>/<segment>/ directories inside).
#
# Each chunk is ~9 GB zipped and expands to about the same size, so write the
# extracted copy somewhere other than a read-only dataset mount. Re-runs are
# idempotent: unzip -n skips files that already exist.
#
# Usage: extract.sh <ZIP_DIR> <EXTRACTED_PATH>
#   ZIP_DIR         directory holding the downloaded Chunk_*.zip files (directly,
#                   or in raw_data/ as the Hugging Face download leaves them);
#                   "" when EXTRACTED_PATH is already extracted (nothing to do)
#   EXTRACTED_PATH  output directory
set -euo pipefail

ZIP_DIR="${1?usage: extract.sh <ZIP_DIR> <EXTRACTED_PATH>}"
EXTRACTED_PATH="${2:?usage: extract.sh <ZIP_DIR> <EXTRACTED_PATH>}"
if [[ -z "$ZIP_DIR" ]]; then
    echo "No archives given; using the extracted segments in $EXTRACTED_PATH as is."
    exit 0
fi

shopt -s nullglob
zips=("$ZIP_DIR"/Chunk_*.zip "$ZIP_DIR"/raw_data/Chunk_*.zip)
if [[ ${#zips[@]} -eq 0 ]]; then
    echo "No Chunk_*.zip found in $ZIP_DIR" >&2
    exit 1
fi

mkdir -p "$EXTRACTED_PATH"
for zip in "${zips[@]}"; do
    echo "extracting $(basename "$zip") ..."
    unzip -n -q "$zip" -d "$EXTRACTED_PATH"
done
echo "Done. Extracted ${#zips[@]} chunk(s) into $EXTRACTED_PATH"
