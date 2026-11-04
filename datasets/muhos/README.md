# MUHOS

[MUHOS](https://m3s-kuura.github.io/muhos-dataset/) (M3S Kuura, University of Oulu;
"Towards Collecting Large-Scale Vehicular Sensor Data for Open Access", ICSA-C 2026,
[IEEE](https://ieeexplore.ieee.org/document/11641223)) records a 2019 Toyota RAV4 Hybrid around
Oulu, Finland, in October 2026, in seven continuous drives ("sets", 82 minutes and 47 km in
total): city centre, residential (Rajakylä, ~40 km/h), gravel road (`soratie`, ~30 km/h),
bumpy gravel (~20 km/h), paved roads at 60 km/h and 80 km/h (Alakylantie) and a highway
(~100 km/h). The roof carries a Carnegie Robotics MultiSense S27 (a colour camera between a
rectified stereo pair, 25 Hz), a Hesai OT128 lidar (10 Hz, with an IMU) and a PCAN-GPS
receiver, and the car's CAN bus is logged. Faces and licence plates are blurred. MUHOS is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): cite the paper above when you use it.

| Cameras | LiDAR | HD map | 3D detections | Driving command | Preference trajectories |
|---|---|---|---|---|---|
| 1 front (960×594 wide-angle, ~128° × 84°) | ✅ Hesai OT128 | — | — | — | — |

## What StandardE2E extracts

- **One frame = one lidar sweep** (10 Hz) with the colour image taken during it: the image
  whose exposure falls inside the sweep's 100 ms and lies closest to its middle. The
  frame's timestamp is that exposure time, on the sensors' clock (the image's stamp minus
  14 ms, see *Calibration*). **One segment = one set** (a continuous drive of 7–16 minutes,
  `segment_id` = `highway`, `city_center`, ...).
- **cameras**: the S27 colour camera as `CameraDirection.FRONT`, the raw wide-angle image
  (`is_fisheye`, OpenCV Kannala-Brandt distortion `D`) with intrinsics and
  `T_ego_from_camera` extrinsics **that StandardE2E estimated from the data**, because MUHOS
  ships none (see *Calibration* below).
- **lidar**: the sweep in the **ego frame**: the lidar frame levelled (z along the road
  normal, x along the direction of travel) with its origin on the road directly below the
  lidar. Every point is moved from its own capture time to the frame time with the ego
  trajectory (the release is not motion-compensated; at 100 km/h a sweep spans 2.8 m of
  travel), and returns from missing echoes and from the car's own body are dropped.
- **global_position** / `aux_data["pose_matrix"]`: the ego pose, integrated from the CAN
  wheel speed and yaw rate and anchored to the GPS, on the sensors' clock (see *Ego
  trajectory*), in a per-set local east/north frame (metres, origin at the set's first GPS
  fix, z = 0, heading counter-clockwise from east), with the speed. The frame's
  latitude/longitude and the original file stamps are in the index (`extra_latitude`,
  `extra_longitude`, `extra_lidar_stamp_ns`, `extra_image_stamp_ns`). `alakylantie_80kmh`
  ships an empty CAN log; there the speed comes from the GPS and the yaw rate from the
  PCAN-GPS gyro.
- **past/future ego trajectories**: from the per-frame poses, via the
  `FuturePastStatesFromMatricesAggregator`.

MUHOS has no annotations, so `frame_detections_3d` is always empty. Not ingested: the
stereo pair (StandardE2E has no stereo slot yet; it is what the camera calibration is
estimated from), the stereo depth and cost images (see *Known issues*), the IMUs (the
lidar's aligns the vehicle-bus clock, see *Ego trajectory*) and the other CAN signals.

## Calibration

The recording kept the image and point topics but not `camera_info` or `tf`, so MUHOS
publishes no camera intrinsics and no camera–lidar transform. StandardE2E estimates them
from the released data and stores them in
[`_muhos_calibration.py`](../../standard_e2e/caching/src_datasets/muhos/_muhos_calibration.py).
They are an estimate, not the authors' calibration:

- **Lidar → ego**: RANSAC ground-plane fits on 200 sweeps of each set give the road normal
  in the lidar frame: the lidar is pitched 6.7° nose-down (6.53–6.94° per set, median of
  the seven) and sits 2.00 m above the road. The direction of travel comes from
  sweep-to-sweep ICP on straight road.
- **Stereo pair**: at standstill, lidar depth × SGBM disparity gives fx·B = 158.3 px·m; with
  the S27's nominal 0.27 m baseline (a rigid fit of the stereo point cloud to the lidar is
  best at 0.265–0.27 m) that is fx = 586 px. The stereo → lidar transform is a rigid
  point-to-plane ICP of the metric stereo point cloud onto the lidar at standstill, where
  neither motion nor timing play a role; the `city_center` and `60kmh` sets agree within
  0.07° and 5 mm.
- **Colour camera**: at standstill, the colour image is warped into the left camera's view
  through the lens model, using every left pixel's stereo depth; dense optical flow
  measures what is left, and the lens model (Kannala-Brandt, square pixels) and its pose
  relative to the left camera are refit. On held-out frames the residual is 0.15 px
  (median). The colour camera sits 3.4 cm right of the left camera, parallel to it within
  0.1°; f = 409 px. As an independent check, strong lidar silhouettes projected into the
  image lie on average 1.8 px from the nearest image edge up to 20° off-axis, 2.2 px at
  20–45°, 2.6 px at 45–55° and 3.1 px at 55–70° (a proxy: not every silhouette point has
  an edge under it, so it stays above zero even for a perfect calibration).
- **Timing**: the cameras expose an image 14 ms (±2 ms) before its stamp, against the
  lidar's per-point times: on the highway, stereo point clouds match the lidar moved to
  that time at every speed (the mismatch at the stamp grows linearly with speed, to 41 cm
  at 95 km/h).

The rig did not move between drives: at low speed the car's own body sits at the same
pixels (within 0.4 px) and lidar coordinates (within 4 mm) in all seven sets, recorded on
1, 7 and 19 October, so one calibration covers them (at highway speed the roof rack tilts,
see *Known issues*). The lens model is fitted inside the stereo
pair's field of view (up to ~45° off-axis); beyond it, it is checked only by the lidar
silhouettes above.

## Ego trajectory

MUHOS has no pose product. Measured on the data (all seven sets):

- **The vehicle-bus logs run on their own clock.** The CAN and GPS logs (one clock) stamp
  an event 0.3–1.3 s before the lidar and the cameras record it, and the gap drifts by
  60–680 ppm within a set (mid-drive: `60kmh` −0.32 s, `city_center` −0.46 s, `highway`
  −0.56 s, `rajakyla` −0.69 s, `alakylantie_80kmh` −0.81 s, `soratie` and `bumpy_gravel`
  −1.06 s). StandardE2E measures it per set by cross-correlating the bus yaw rate with the
  lidar IMU's gyro (sensor clock) in 60 s windows (correlation 0.97–0.999 over whole
  drives), fits offset + drift, and maps every bus timestamp onto the sensor clock.
  Afterwards, lidar odometry on `city_center`, `soratie` and `bumpy_gravel` matches the
  trajectory's yaw rate through turns within 20 ms and its speed through departures within
  50 ms (they were 0.45–1.26 s off).
- The GPS **fixes at 1 Hz**: position, course and speed change once a second, and the
  10 Hz CAN messages repeat the last fix.
- A fix's **position** reaches the bus logger **about 0.2 s** after the moment it
  describes: the speed derived from consecutive positions matches the CAN wheel speed
  best when shifted by 0.20–0.26 s (five sets; the highway's near-constant speed does not
  constrain it). The fix's reported **speed and course lag much more, 0.6–1.5 s**, by a
  different amount in each set, so StandardE2E does not use the GPS speed and uses the
  course only on straight stretches, where a lag does not change it.
- The CAN wheel speed (`SPEED`, ~38 Hz) reads 1.2–1.7 % below the distance between GPS
  fixes, per set. The yaw rate (`KINEMATICS.YAW_RATE`, 10 Hz) is quantised to 0.244 deg/s,
  its scale agrees with the GPS course change across 27 turns (median ratio 1.008), and
  it reads −0.07 or +0.17 deg/s (one quantum) at standstill.

So the pose is dead-reckoned from the scaled wheel speed and the bias-corrected yaw rate,
then pulled towards the GPS by corrections that vary slowly (Gaussian, σ = 15 s): the
position towards the lag-corrected fixes and the heading towards the course on straight
stretches. Seconds-scale motion (the past/future trajectories) comes from the vehicle bus;
absolute position and drift come from the GPS. Per set, the trajectory stays within
1.0–1.5 m (median) of the GPS fixes, the GPS's own noise. Against lidar odometry
(point-to-plane ICP between consecutive sweeps, chained) on the highway at ~95 km/h, the
trajectory's motion over 1 s (26 m) agrees to 4 cm along-track and 3 cm laterally
(median; 16 cm and 32 cm at the 95th percentile) and its heading change to 0.17° (95th
percentile); its distance scale is within 0.25 %.

`alakylantie_80kmh` has no vehicle-bus data, so its speed is the distance between GPS fixes
and its yaw rate the PCAN-GPS gyro, which reads 8.7 % low against the GPS course
(27 turns) and reaches the log 47 ms later than the CAN yaw rate (40–59 ms on the six other
sets); both are corrected. Run the same way on the six other sets, this fallback finds the
bus clock within 12 ms of the CAN-based estimate and ends 3 s futures within 0.45–0.69 m
(median) of the CAN-based trajectory (1.2 m on bumpy gravel, where the car is slow).

## Known issues

- **No calibration in the release** (source data): see *Calibration*. The colour
  camera's lens model is extrapolated beyond ~45° off-axis.
- **The vehicle-bus clock is 0.3–1.3 s off the sensors' and drifts** (source data): used as
  stamped, the CAN/GPS-derived poses and past/future trajectories would lag the images
  and the lidar by up to a second. StandardE2E aligns the clocks per set with the lidar's
  IMU (see *Ego trajectory*), so the `imu` stream must be downloaded.
- **Images are stamped 14 ms after their exposure** (source data): StandardE2E uses the
  exposure time (see *Calibration*); at 100 km/h the car moves 0.4 m in 14 ms.
- **The sensor rack pitches at highway speed** (source data): above ~60 km/h the car's own
  body moves up to 2 px in the image and its returns up to 2 cm in the lidar, consistent
  with the roof rack tilting ~0.3° nose-up in the wind; the camera–lidar geometry does not
  change (stereo point clouds fit the lidar at every speed). The ego frame is levelled
  with one constant, so at ~100 km/h the road ahead can sit ~0.2–0.3° off level (about
  15–20 cm at 40 m).
- **GPS at 1 Hz, repeated at 10 Hz on the bus, its speed and course late** (source data):
  using the 10 Hz GPS messages as they are gives a stair-stepped track, and the reported
  speed and course trail the position by another 0.4–1.3 s. StandardE2E uses only the fix
  updates, as described in *Ego trajectory*.
- **`alakylantie_80kmh` has an empty CAN log** (source data): its
  `toyota_CAN_20261001_144823.json` is `[]`, so StandardE2E falls back to GPS and the
  PCAN-GPS gyro there (see *Ego trajectory*).
- **Streams cover slightly different spans, and the colour camera drops frames** (source
  data): the CAN log ends 10 s before the lidar in `city_center` (and the GPS starts 3 s
  after it in `alakylantie_80kmh`); sweeps outside the bus log are skipped. Each camera
  folder also holds one stray image from minutes before the drive. The colour camera
  delivers 86–99 % of its 25 Hz frames; the 0.2–0.5 % of sweeps without an image are
  skipped.
- **The depth images are not metric** (source data): the export normalised each
  `/left/depth` image to its own min–max range and saved it as a TURBO colour map, so depth
  cannot be recovered from the PNGs. Not ingested.
- **The lidar is not motion-compensated and includes the car body** (source data): the
  points of a sweep span 100 ms, and about 1,700 points per sweep are the car's hood, the
  roof sign and the roof. StandardE2E compensates the motion with the per-point times and
  drops points inside the car's box.
- **Release naming is not uniform** (source data): camera archives carry an
  `_anonymized` suffix except in the two gravel sets, and CAN/GPS logs are named by start
  time (`toyota_CAN_20261007_131926.json`), by set (`toyota_CAN_city_center.json`) or
  otherwise (`toyota_CAN_60kmtie.json`). The website's set table omits the city-centre
  drive and lists thermal (FLIR) cameras that are not in the release.
- **Archive members are not in time order** (source data): a partially downloaded or
  extracted archive holds a random subset of the drive, not its beginning. The pipeline's
  download step checks each archive's size.

## Environment

This folder is its own [uv](https://docs.astral.sh/uv/) project:
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) pin StandardE2E (this
checkout, editable) plus `laspy[lazrs]` (the `.laz` point clouds) and `pyproj` (GPS to
metres), without StandardE2E's training-only dependencies (PyTorch), which
preprocessing never imports. MUHOS has no SDK. [`.python-version`](.python-version)
pins Python 3.12 (`requires-python` allows 3.12–3.14). stepix builds the environment into
`.venv/` here on first use; to run a command in it yourself:
`uv run --project datasets/muhos <command>`.

## Pipeline

Set the paths below, then run, from the repository root:

```bash
uvx stepix run datasets/muhos/README.md
```

`download = true` fetches the archives first (step 1); without it, put each set's
archives in `tar_dir/imu_<set>_processed/` yourself (the download links are on the
[dataset page](https://m3s-kuura.github.io/muhos-dataset/)). `sets` selects the drives and
`streams` the archives per drive. The default streams are what StandardE2E reads: the
lidar, the colour camera, the CAN/GPS logs and the IMUs (the lidar's aligns the clocks),
about half of a set's size: 142 GB of the release's 293 GB. `debug = true` processes
only the first set, as a smoke test.

```stepix
[params]
tar_dir = "/path/to/muhos_archives"           # holds imu_<set>_processed/*.tar; "" if extracted
extracted_path = "/path/to/muhos"
output_path = "/path/to/standard_e2e_data"
config = "../../configs/muhos.yaml"
sets = "city_center rajakyla soratie bumpy_gravel 60kmh alakylantie_80kmh highway"
streams = "lidar_points aux_image_color can imu"  # also: left_image_rect right_image_rect left_depth left_cost
num_workers = 8
download = false                              # true: fetch the data first (step 1)
debug = false
```

### 1. Download (optional)

With `download = true`, this step fetches each selected set's archives from
[a3s.fi](https://a3s.fi) with `curl` into `tar_dir/imu_<set>_processed/`. No account is
needed; finished archives are skipped and an interrupted download resumes.

```stepix
[steps.download]
env = "."
run = '''
if [ {params.download} != true ]; then
    echo "download = false: nothing to download."
    exit 0
fi
if [ -z {params.tar_dir} ]; then
    echo "download = true needs tar_dir: the folder to download into." >&2
    exit 1
fi
bash download.sh {params.tar_dir} {params.sets} {params.streams}
'''
cache = true
```

### 2. Extract the archives

[`extract.sh`](extract.sh) unpacks the selected streams of the selected sets into
`extracted_path/imu_<set>_processed/`; archives extracted before are skipped. Set
`tar_dir = ""` if `extracted_path` already holds the extracted sets.

```stepix
[steps.extract]
env = "."
run = '''
if [ -z {params.tar_dir} ]; then
    echo "tar_dir is empty: using the extracted sets in" {params.extracted_path}
    exit 0
fi
bash extract.sh {params.tar_dir} {params.extracted_path} {params.sets} {params.streams}
'''
requires = ["download"]
cache = true
```

### 3. Check the input

Each set needs its lidar sweeps, colour images, CAN/GPS logs and the lidar's IMU log.

```stepix
[steps.check]
env = "."
run = '''
status=0
for set in $(printf '%s' {params.sets}); do
    d={params.extracted_path}/imu_${set}_processed
    n_lidar=$(find "$d/lidar_points" -name "*.laz" 2>/dev/null | wc -l)
    n_image=$(find "$d" -mindepth 2 -maxdepth 2 -path "$d/aux_image_color*" -name "*.png" 2>/dev/null | wc -l)
    n_can=$(ls "$d"/can/toyota_CAN_*.json "$d"/can/toyota_GPS_*.json 2>/dev/null | wc -l)
    echo "$set: $n_lidar sweeps, $n_image colour images, $n_can CAN/GPS logs"
    if [ "$n_lidar" -eq 0 ] || [ "$n_image" -eq 0 ] || [ "$n_can" -ne 2 ] || [ ! -f "$d/imu/lidar_imu.json" ]; then
        echo "  missing lidar_points/, aux_image_color*/, can/toyota_{CAN,GPS}_*.json or imu/lidar_imu.json in $d" >&2
        status=1
    fi
done
exit $status
'''
requires = ["extract"]
cache = true
```

### 4. Preprocess

Writes `<output_path>/muhos/all/` (one `.npz` per frame plus `index.parquet`). MUHOS has
no official split, so the split is the passthrough label `all`; every set in
`extracted_path` is processed.

```stepix
[steps.preprocess]
env = "."
run = '''
STANDARD_E2E_DEBUG={params.debug} python -m standard_e2e.caching.process_source_dataset muhos \
    --input_path={params.extracted_path} \
    --output_path={params.output_path} \
    --split=all \
    --num_workers={params.num_workers} \
    --config_file={params.config} \
    --do_parallel_processing
'''
requires = ["check"]
cache = true
```
