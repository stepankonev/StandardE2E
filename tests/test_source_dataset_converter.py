"""Tests for the frame pool's worker setup in SourceDatasetConverter."""

import cv2
from threadpoolctl import threadpool_info, threadpool_limits

from standard_e2e.caching import source_dataset_converter as converter


def test_init_worker_stores_the_processor_and_runs_libraries_single_threaded():
    processor = object()
    opencv_threads = cv2.getNumThreads()
    with threadpool_limits():  # puts the original BLAS limits back on exit
        converter._init_worker(processor)  # type: ignore[arg-type]
        blas_threads = [
            p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"
        ]
        assert converter._WORKER_PROCESSOR is processor
        assert cv2.getNumThreads() == 1
        assert blas_threads and all(n == 1 for n in blas_threads)
    cv2.setNumThreads(opencv_threads)
    converter._WORKER_PROCESSOR = None
