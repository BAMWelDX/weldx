"""PyTest configuration."""

import pytest

collect_ignore_glob = [
    "weldx/visualization/*.py",
]


@pytest.fixture(autouse=True)
def mock_rw_buffer_weldxfile(request, monkeypatch):
    if not request.config.getoption("--weldx-file-rw-buffer"):
        return

    monkeypatch.setattr("weldx.asdf.util._USE_WELDX_FILE", True)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--weldx-file-rw-buffer",
        action="store_true",
        default=False,
        help="the read/write buffer functions use WeldxFile internally",
    )
    parser.addoption(
        "--weldx-file-rw-buffer-disp-header",
        action="store_true",
        default=False,
        help="invoke display header to provoke side effects",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
