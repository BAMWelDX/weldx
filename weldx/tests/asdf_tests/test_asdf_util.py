"""tests for asdf utility functions."""
import io

import pytest

from weldx.asdf.cli.welding_schema import single_pass_weld_example
from weldx.asdf.util import get_yaml_header


@pytest.fixture(scope="function")
def create_file_and_buffer():
    """Create a temporary named file AND a buffer of its contents.

    For single pass welding example
    """
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".asdf", delete=False) as ntf:
        ntf.close()
        single_pass_weld_example(out_file=ntf.name)
        with open(ntf.name, "rb") as fh:
            buffer = io.BytesIO(fh.read())
        # now run tests
        yield ntf.name, buffer
        # remove real file
        os.remove(ntf.name)


@pytest.mark.parametrize(
    argnames=["index", "parse"],
    argvalues=[
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ],
)
def test_get_yaml_header(create_file_and_buffer, index, parse):
    """Check that get_yaml_header can read from BytesIO and file name.

    Also checks that the parse argument is returning the parsed header, when requested.
    """
    file = create_file_and_buffer[index]
    header = get_yaml_header(file, parse)
    if parse:
        assert isinstance(header, dict)
        assert header["asdf_library"]
    else:
        assert isinstance(header, str)
        assert "asdf_library" in header
