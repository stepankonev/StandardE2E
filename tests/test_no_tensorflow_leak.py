"""Regression guard for PR 2 step 2.6.

After ``import standard_e2e`` (the base public surface), ``tensorflow``
must NOT appear in ``sys.modules``: per ADR 0005, TF is a Waymo-only
extra, and base-install users (no ``[waymo]``) should be able to
import the framework without TF on disk at all.

This test runs in a subprocess so that previously-imported TF (pulled
in by other tests in this run) does not contaminate ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys


def test_base_import_does_not_pull_tensorflow():
    code = (
        "import sys\n"
        "import standard_e2e\n"
        "import standard_e2e.unified_dataset\n"
        "import standard_e2e.data_structures\n"
        "import standard_e2e.dataset_utils.frame_loader\n"
        # caching/__init__.py exports SourceDatasetConverter +
        # TFRecSourceDatasetConverter; importing them must not pull TF.
        "import standard_e2e.caching\n"
        "assert 'tensorflow' not in sys.modules, "
        "'tensorflow leaked into sys.modules after base imports'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"base import smoke failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
