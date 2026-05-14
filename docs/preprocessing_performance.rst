Preprocessing Performance
=========================

This page describes how the source-dataset preprocessing pipeline scales,
what the per-dataset throughput characteristics look like, and the knobs
available for tuning. Numbers below were measured on a 32-core box with
HDD input and NVMe output; absolute rates will vary with hardware, but
the relative shape of each dataset should be similar.

Frame-stage rates (par-32, full-modality chain, cold page cache)
----------------------------------------------------------------

The frame-stage runs through every adapter the user has configured. The
table is for the production ``configs/full_all_modalities.yaml`` chain
(cameras + lidar + HD-map BEV raster + 3D detection identity adapter +
identity-only adapters for past/future/intent/preference).

+---------------------+--------------+----------------+-----------------------+
| dataset             | fr/s par-32  | full split est.| notes                 |
+=====================+==============+================+=======================+
| ``waymo_e2e``       | ~175         | ~1.1 h         | forkserver pool       |
+---------------------+--------------+----------------+-----------------------+
| ``waymo_perception``| ~50          | ~1.1 h         | forkserver pool;      |
|                     | (steady)     |                | disk-spilled HD-map   |
|                     |              |                | cache + numpy lidar   |
+---------------------+--------------+----------------+-----------------------+
| ``av2_sensor``      | ~55          | ~37 min        | fork pool             |
+---------------------+--------------+----------------+-----------------------+
| ``av2_lidar``       | ~190         | ~6.0 h         | fork pool;            |
|                     |              |                | 16 000 logs           |
+---------------------+--------------+----------------+-----------------------+
| ``navsim``          | ~50          | ~3.0 h         | fork pool             |
+---------------------+--------------+----------------+-----------------------+

``waymo_perception`` steady-state is the rate after fixed setup costs
amortize (forkserver pool startup + HD-map prescan).  Smoke runs on
small slices (1-4 tfrecords) will report lower aggregate rates because
fixed costs dominate at that scale.

Measured full-split run (Waymo Perception training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validated with an end-to-end run of the production CLI on the full
training split (798 tfrecords, 158 081 frames), par-32, full-modality
chain, cold page cache. Wall: **89 min 37 s** end-to-end.

==========================================  ============  ===============
phase                                        seconds       fraction
==========================================  ============  ===============
HD-map prescan (parallel, 8 threads)         ~16           0.3%
Pool startup (forkserver + initializer)      ~5            0.1%
**Frame stage** (158 081 frames @ ~41 fr/s)  **~3 860**    **71.8%**
``FuturePastStatesFromMatricesAggregator``   ~90           1.7%
``FutureDetectionsAggregator``               ~1 400        26.0%
Index + teardown                             ~6            0.1%
**Total wall**                               **5 377 s**   **100%**
==========================================  ============  ===============

Frame-stage steady-state ran at ~46–54 fr/s through the bulk of the
run (only the very last few large logs slow it down as workers
finish). The disk-spilled HD-map cache and the numpy lidar decode are
the reason this stage stays >40 fr/s sustained at par-32; on the
original ``main`` code path the same chain ran at ~0.57 fr/s,
projecting to ~76 h for these 158 081 frames — so the new code path is
**~50× faster** at the frame stage on this split.

The ``FutureDetectionsAggregator`` is now the single biggest single
cost in the pipeline — ~23 min on its own, vs ~64 min for the frame
stage. It rasterises per-frame future-detection trajectories from
segment-level sequences; that's a separate stage and would be an
obvious next target for optimisation.

Output footprint for the full split:

- ``.npz`` frame files: **19 GB** total (~120 KB/frame after the
  aggregators add per-frame future/past state arrays; ~44 KB/frame
  before they run).
- ``_map_cache/`` (intermediate prescan scratch, safe to delete after
  preprocessing): 186 MB.
- ``index.parquet``: 4.2 MB.

What makes it fast
------------------

Several design decisions cooperate to keep workers saturated and avoid
unnecessary work.

**1. Per-dataset multiprocessing start method.** ``multiprocessing.Pool``
is hard-coded to ``"spawn"`` by upstream Python; spawn re-imports the
full module tree on every worker (~5 s per worker, dominated by the
TensorFlow import). StandardE2E selects a start method per dataset:

- AV2 Sensor / AV2 LiDAR / NAVSIM → ``"fork"`` (their worker hot path
  is TF-free, so fork is safe and instantaneous).
- Waymo Perception / Waymo E2E → ``"forkserver"`` (a single helper
  process imports TF once and forks workers from it before any TF op
  fires; sidesteps the fork-after-TF-init deadlock without paying the
  spawn import tax 32 times).

A 32-worker pool on Waymo now starts in ~5 s instead of ~150 s.

**2. Lazy modality loading.** Each adapter declares
:attr:`~standard_e2e.caching.adapters.abstract_adapter.AbstractAdapter.consumes_attrs`
— the set of :class:`~standard_e2e.data_structures.frame_data.StandardFrameData`
fields it reads. The processor unions these across the configured chain;
modality builds (cameras, lidar, HD-map, detections) are gated on
``needs_attr(name)``. A cameras-only chain skips the lidar TF decode;
a lidar-only chain skips JPEG decoding and the HD-map raster.

**3. Pool dispatch via initializer.** ``Pool.imap`` pickles its task
function on every dispatch. For Waymo Perception that bound method
carried a ~200 MB HD-map prescan cache, capping throughput at ~0.6 fr/s
regardless of worker count. StandardE2E ships the processor once per
worker via ``Pool.initializer`` and uses a module-level worker function
for per-task dispatch.

**4. Disk-spilled HD-map prescan cache.** The Waymo Perception
prescan reads frame 0 of every tfrecord to build a per-segment cache of
``map_features``. Rather than holding the ~200 MB cache in the parent
(and pickling it to every worker), the prescan writes per-segment
``.pkl`` files under ``<output_path>/_map_cache/``. Each worker
lazy-loads only the segments it actually touches (~6 MB per worker
vs ~200 MB).

**5. Pure-numpy Waymo lidar decode.** Upstream
``frame_utils.convert_range_image_to_point_cloud`` and
``parse_range_image_and_camera_projection`` both call into the TF
runtime — eager-mode kernel dispatch, intra-op threading, ``tf.gather_nd``,
``tf.linalg.inv``. Under a 32-process worker pool the cumulative
runtime overhead makes each frame ~485 ms in workers, dominating the
pipeline. The numpy replacements in
:mod:`standard_e2e.utils.waymo_lidar_numpy` do the same work without
ever entering the TF runtime in a worker:

- :func:`~standard_e2e.utils.waymo_lidar_numpy.numpy_parse_range_image_and_camera_projection`
  uses ``zlib.decompress`` + proto parse (no ``tf.io.decode_compressed``).
- :func:`~standard_e2e.utils.waymo_lidar_numpy.numpy_convert_range_image_to_point_cloud`
  builds the spherical-to-cartesian transform, applies each laser's
  extrinsic, and (for the TOP laser only) applies the per-pixel pose
  correction and the frame-pose inverse — all via plain ndarray ops.

Specific micro-optimisations:

- Proto-to-ndarray conversion goes through ``np.fromiter`` with a
  preallocated count, which is ~1.7× faster than ``np.array(container)``
  on the large ``RepeatedScalarContainer`` instances that come out of
  ``MatrixFloat.ParseFromString``. On a typical frame this alone saves
  ~150 ms of Python iteration.
- Beam-inclination reconstruction matches upstream's pixel-centre
  formula ``inc[i] = ((i + 0.5) / H) * (max - min) + min`` (NOT
  ``np.linspace``, which would offset by half a row and produce
  ~7 cm of error per side laser — caught during equivalence checks).

End-to-end this gives ~2× faster lidar work in isolation, and ~4× more
worker throughput at par-32 (workers no longer serialize on the TF
runtime). Equivalence with the upstream TF implementation is verified
by ``tests/test_waymo_lidar_numpy.py``:

- ``_rotation_from_rpy`` matches a manual ``R_yaw @ R_pitch @ R_roll``
  composition exactly.
- ``_proto_floats_to_ndarray`` preserves values and dtype.
- End-to-end ``numpy_convert_range_image_to_point_cloud`` on a synthetic
  multi-laser frame matches ``frame_utils.convert_range_image_to_point_cloud``:
  side lasers bit-exact (``atol=1e-5``), TOP laser within ``atol=1e-2``
  (the ~3 mm float32 drift from differing ``inv`` implementations is
  well below typical Waymo lidar sensor noise).

These tests run on every commit and guard against silent regressions if
the numpy path drifts away from the canonical TF output.

**6. Pre-listed AV2 / NAVSIM iterators.** AV2 and NAVSIM converters
materialize the full ``(log, frame)`` tuple list at converter-init time
instead of globbing each log's directory lazily during iteration. On
HDD this consolidates many in-flight directory scans into one bulk pass
at startup so they don't interleave with workers' per-frame reads.

**7. Parallel HD-map prescan.** Waymo Perception's per-file prescan
runs in an 8-thread ``ThreadPoolExecutor`` (configurable via
``WP_PRESCAN_THREADS``). Python releases the GIL during the blocking
syscall, so concurrent file opens / reads are served by the kernel's
NCQ. Cache writes stay on the main thread to avoid races on the cache
file map.

Tuning knobs
------------

``--num_workers``
   Worker pool size. Defaults to ``$(nproc)`` in the CLI. Per-dataset
   max_workers can cap this internally (currently no dataset caps
   below 32; Waymo Perception's earlier par-8 cap was lifted after the
   disk-cache change). Useful to lower in cgroup-constrained
   environments.

``WP_PRESCAN_THREADS`` (env)
   Threads used by the parallel HD-map prescan on Waymo Perception.
   Default ``8``. Set to ``1`` for sequential behavior; raise on
   fast-NVMe inputs.

``STANDARD_E2E_DEBUG=true``
   Limit each dataset to one log / tfrecord. Quick smoke; some
   datasets may truncate segment continuity in this mode.

The shape of a smoke run
------------------------

For small N (e.g. 4 tfrecords on Waymo Perception), fixed costs are a
large fraction of total wall time. A representative breakdown:

- ``T_init`` — converter ``__init__`` (HD-map prescan on Waymo
  Perception; pre-listing iterators on AV2 / NAVSIM): seconds to a few
  tens of seconds, scales with file count, runs only when an adapter
  consumes ``hd_map``.
- ``T_pool`` — pool startup + first frame: ~5–10 s on forkserver,
  near-instant on fork.
- ``T_steady`` — frame-stage processing: the rate cited in the table
  above.
- ``T_teardown`` — pool close + index parquet write: sub-second.

For projections at full-split scale, only ``T_steady`` matters in
practice — the others amortize over hundreds of thousands of frames.

Output equivalence
------------------

All optimizations preserve frame outputs at float-tolerant equivalence
(``atol=1e-3``). The only known non-bit-exact path is the Waymo
Perception TOP-laser numpy decode, which differs by ≤3 mm from the TF
implementation due to ``np.linalg.inv`` versus ``tf.linalg.inv``
internal differences — below typical Waymo lidar sensor noise.
