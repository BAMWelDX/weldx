import jsonschema
import pandas as pd
import pytest

from weldx.asdf.tags.weldx.debug.test_property_tag import PropertyTagTestClass

from .utility import _write_read_buffer


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
