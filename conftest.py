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
