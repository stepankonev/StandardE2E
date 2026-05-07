Caching Segment Context
=======================

Utilities for aggregating segment-level context that supplements frame-level data.

Aggregators run **after** the per-frame conversion stage and operate one
segment at a time, reading the segment's npz files, computing the
context (past / future trajectories, aggregated detection trajectories,
…), and writing each frame back in place. Segments are independent —
disjoint files, no shared aggregator state — so the per-segment loop
fans out across a ``multiprocessing.Pool`` controlled by the same
``--num_workers`` and ``--do_parallel_processing`` flags as the frame
stage. The default start method is ``"spawn"`` because the parent
process imports TensorFlow / OpenCV at module load and ``fork`` would
inherit that state into workers.

.. note::

   Custom :class:`~standard_e2e.caching.segment_context.SegmentContextAggregator`
   subclasses must be picklable: a ``spawn`` worker re-imports the
   class by qualname, so the class needs to live in an installed
   package (not an ad-hoc test module) and avoid carrying
   un-picklable state (open file handles, locks, GPU contexts). The
   escape hatch is to call ``process(..., do_parallel=False)`` for a
   sequential fall-back.

.. autosummary::
   :toctree: generated/caching
   :nosignatures:

   standard_e2e.caching.segment_context.SegmentContextAggregator
   standard_e2e.caching.segment_context.FuturePastStatesFromMatricesAggregator
   standard_e2e.caching.segment_context.FutureDetectionsAggregator
