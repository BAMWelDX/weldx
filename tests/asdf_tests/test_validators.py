from io import BytesIO

import asdf
import jsonschema
import pandas as pd
import pytest

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.tags.weldx.debug.test_property_tag import PropertyTagTestClass


def _write_read_buffer(tree):
    # Write the data to buffer
    with asdf.AsdfFile(
        tree,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        ignore_version_mismatch=False,
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff, all_array_storage="inline")
        buff.seek(0)

    # read back data from ASDF file contents in buffer
    with asdf.open(
        buff, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
    ) as af:
        data = af.tree
    return data, buff


def test_property_tag_validator():
    """Test custom ASDF shape validators."""
    test = PropertyTagTestClass()
    tree = {"root_node": test}
    _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = PropertyTagTestClass(prop3=pd.Timedelta(2, "s"))
        tree = {"root_node": test}
        _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = PropertyTagTestClass(prop3="STRING")
        tree = {"root_node": test}
        _write_read_buffer(tree)
