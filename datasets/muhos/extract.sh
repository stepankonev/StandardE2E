#!/usr/bin/env bash
# Extract MUHOS archives: extract.sh <tar_dir> <extracted_path> "<sets>" "<streams>"
# For every set and stream, <tar_dir>/imu_<set>_processed/<stream>[_anonymized].tar is
# unpacked into <extracted_path>/imu_<set>_processed/; an archive extracted before (marker
# file) is skipped.
set -eu
tar_dir=$1
extracted_path=$2
read -r -a sets <<< "$3"
read -r -a streams <<< "$4"
for set in "${sets[@]}"; do
    bucket=imu_${set}_processed
    mkdir -p "$extracted_path/$bucket"
    for stream in "${streams[@]}"; do
        archive=""
        for name in "${stream}_anonymized" "$stream"; do
            if [ -f "$tar_dir/$bucket/$name.tar" ]; then archive=$tar_dir/$bucket/$name.tar; break; fi
        done
        if [ -z "$archive" ]; then
            echo "No archive for stream '$stream' in $tar_dir/$bucket/." >&2
            exit 1
        fi
        marker=$extracted_path/$bucket/.$(basename "$archive").extracted
        if [ -e "$marker" ]; then
            echo "already extracted: $bucket/$(basename "$archive")"
            continue
        fi
        echo "extracting $bucket/$(basename "$archive")"
        tar -xf "$archive" -C "$extracted_path/$bucket"
        touch "$marker"
    done
done
