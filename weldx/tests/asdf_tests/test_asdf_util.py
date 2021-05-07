"""tests for asdf utility functions."""
import pytest

from weldx import WeldxFile
from weldx.asdf.util import get_yaml_header


@pytest.fixture(scope="function")
def create_file_and_buffer(tmpdir):
    """Create a temporary named file AND a buffer of its contents."""
    import tempfile

    name = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
    with WeldxFile(name, mode="rw") as fh:
        fh["some_attr"] = True
        buffer = fh.write_to()
    # now run tests
    yield name, buffer


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
