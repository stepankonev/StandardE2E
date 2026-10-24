#!/usr/bin/env bash
# Download MUHOS archives from a3s.fi (public, no account): download.sh <tar_dir> "<sets>" "<streams>"
# Each archive lands in <tar_dir>/imu_<set>_processed/<stream>[_anonymized].tar; finished
# archives are skipped and interrupted ones resume.
set -u
tar_dir=$1
read -r -a sets <<< "$2"
read -r -a streams <<< "$3"
command -v curl > /dev/null || { echo "download = true needs curl." >&2; exit 1; }
base=https://a3s.fi/swift/v1
status=0
for set in "${sets[@]}"; do
    bucket=imu_${set}_processed
    mkdir -p "$tar_dir/$bucket"
    for stream in "${streams[@]}"; do
        # Camera archives carry an _anonymized suffix (faces and plates blurred) in every
        # set except the two gravel ones.
        found=""
        for name in "${stream}_anonymized" "$stream"; do
            size=$(curl -sfI -m 30 "$base/$bucket/$name.tar" | awk -F': ' 'tolower($1)=="content-length" {print $2}' | tr -d '\r')
            if [ -n "$size" ]; then found=$name; break; fi
        done
        if [ -z "$found" ]; then
            echo "No archive for stream '$stream' in set '$set' ($base/$bucket/)." >&2
            status=1
            continue
        fi
        out=$tar_dir/$bucket/$found.tar
        have=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [ "$have" = "$size" ]; then
            echo "have $bucket/$found.tar"
            continue
        fi
        echo "downloading $bucket/$found.tar ($((size / 1000000)) MB)"
        for _ in $(seq 1 30); do
            curl -sS -L -C - --speed-limit 10000 --speed-time 60 -o "$out" "$base/$bucket/$found.tar" && break
            sleep 10
        done
        have=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [ "$have" != "$size" ]; then
            echo "Incomplete: $out ($have of $size bytes); rerun to resume." >&2
            status=1
        fi
    done
done
exit $status
