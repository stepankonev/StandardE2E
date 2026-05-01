"""Auto-mark every test under ``tests/integration/`` as ``waymo`` + ``integration``.

Integration tests on this repo all touch real Waymo source data on
disk and / or import the Waymo proto stack, so they require the
``[waymo]`` extra. Marking them via this conftest avoids per-file
``pytestmark`` boilerplate and makes ``pytest -m 'not waymo'`` a clean
filter for the base CI job.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/integration" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.waymo)
            item.add_marker(pytest.mark.integration)
