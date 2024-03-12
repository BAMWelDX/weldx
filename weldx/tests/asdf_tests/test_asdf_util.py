"""tests for asdf utility functions."""

from __future__ import annotations

import io
import unittest
from dataclasses import dataclass

import numpy as np
import pytest

from weldx.asdf.file import WeldxFile
from weldx.asdf.util import (
    _ProtectedViewDict,
    dataclass_serialization_class,
    get_highest_tag_version,
    get_schema_tree,
    get_yaml_header,
)


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


def test_get_yaml_header_win_eol():
    """Ensure we can read win and unix line endings with get_yaml_header."""
    wx = WeldxFile(tree=dict(x=np.arange(20)), mode="rw")  # has binary blocks
    buff = wx.write_to()
    native = buff.read()  # could be "\r\n" or "\n"
    unix = native.replace(b"\r\n", b"\n")
    win = unix.replace(b"\n", b"\r\n")
    for b in [native, unix, win]:
        get_yaml_header(io.BytesIO(b))


def _to_yaml_tree_mod(tree):
    tree["a"] += ["d"]
    return tree


def _from_yaml_tree_mod(tree):
    tree["a"] += ["e"]
    return tree


@pytest.mark.parametrize(
    "val_a, exp_val_a_tree, exp_val_a_dc, to_yaml_tree_mod, from_yaml_tree_mod,"
    "sort_string_lists",
    [
        (["c", "b", "a"], ["a", "b", "c"], ["a", "b", "c"], None, None, True),
        (["c", "b", "a"], ["c", "b", "a"], ["c", "b", "a"], None, None, False),
        # not a pure string list -> no sorting
        (["c", 1, "a"], ["c", 1, "a"], ["c", 1, "a"], None, None, True),
        (["c", 1, "a"], ["c", 1, "a"], ["c", 1, "a"], None, None, False),
        # check to_yaml_tree_mod is called
        (["c", "b"], ["b", "c", "d"], ["b", "c", "d"], _to_yaml_tree_mod, None, True),
        (["c", "b"], ["c", "b", "d"], ["c", "b", "d"], _to_yaml_tree_mod, None, False),
        # check_from_yaml_tree_mod is called
        (
            ["c"],
            ["c", "d"],
            ["c", "d", "e"],
            _to_yaml_tree_mod,
            _from_yaml_tree_mod,
            False,
        ),
        (["c"], ["c"], ["c", "e"], None, _from_yaml_tree_mod, False),
    ],
)
def test_dataclass_serialization_class(
    val_a,
    exp_val_a_tree,
    exp_val_a_dc,
    to_yaml_tree_mod,
    from_yaml_tree_mod,
    sort_string_lists,
):
    """Test the `dataclass_serialization_class` function.

    The test defines a dataclass and its corresponding serialization class using
    `dataclass_serialization_class`. It first calls ``to_yaml_tree`` to get tree from
    the generated serialization class. Afterwards the tree is used with the
    ``from_yaml_tree`` method to construct a new dataclass instance. The results of the
    function calls are checked against the expected values.

    Parameters
    ----------
    val_a :
        Initial value of the dataclasses' variable a
    exp_val_a_tree :
        Expected value of the variable a in the tree after `to_yaml_tree` was run
    exp_val_a_dc :
        Expected value of the variable a of the reconstructed dataclass
    to_yaml_tree_mod :
        The value passed as corresponding function parameter
    from_yaml_tree_mod :
        The value passed as corresponding function parameter
    sort_string_lists
        The value passed as corresponding function parameter

    """

    @dataclass
    class _DataClass:
        a: list[str]
        b: int = 1

    dc = _DataClass(a=val_a, b=2)

    dataclass_asdf = dataclass_serialization_class(
        class_type=_DataClass,
        class_name="Test",
        version="0.1.0",
        sort_string_lists=sort_string_lists,
        to_yaml_tree_mod=to_yaml_tree_mod,
        from_yaml_tree_mod=from_yaml_tree_mod,
    )
    tree = dataclass_asdf().to_yaml_tree(dc, None, None)

    assert tree["b"] == 2
    assert tree["a"] == exp_val_a_tree

    dc_restored = dataclass_asdf().from_yaml_tree(tree, None, None)

    assert dc_restored.b == 2
    assert dc_restored.a == exp_val_a_dc


def test_get_highest_tag_version():
    """Test getting some tags from the WeldxExtension."""
    assert (
        get_highest_tag_version("asdf://weldx.bam.de/weldx/tags/uuid-*")
        == "asdf://weldx.bam.de/weldx/tags/uuid-0.1.0"
    )
    assert get_highest_tag_version("asdf://weldx.bam.de/weldx/tags/uuid-1.*") is None

    with pytest.raises(ValueError):
        get_highest_tag_version("asdf://weldx.bam.de/weldx/tags/**-*")


def test_get_schema_tree():
    d = get_schema_tree("single_pass_weld-0.1.0")
    assert isinstance(d, dict)


class TestProtectedView(unittest.TestCase):
    def setUp(self):
        data = dict(foo="blub", bar=42)
        self.protected_key = "bar"
        self.view = _ProtectedViewDict(protected_keys=[self.protected_key], data=data)

    def test_protected_keys_hidden(self):
        with pytest.warns(UserWarning, match="tried to manipulate"):
            assert self.protected_key not in self.view.keys()
            assert self.protected_key not in self.view
            assert (self.protected_key, 42) not in self.view.items()

    def test_allowed_access(self):
        assert self.view["foo"] == "blub"

    def test_illegal_access(self):
        with pytest.warns(UserWarning, match="tried to manipulate"):
            with pytest.raises(KeyError):
                _ = self.view["bar"]

            with pytest.raises(KeyError):
                del self.view["bar"]

    def test_access_non_existent(self):
        with pytest.raises(KeyError):
            _ = self.view["no"]

    def test_update(self):
        expected_match = "manipulate an ASDF internal structure"
        warning_type = UserWarning

        with pytest.warns(warning_type, match=expected_match):
            self.view.update(dict(bar=None), foo=1)
        assert self.view["foo"] == 1

    def test_popitem(self):
        k, v = self.view.popitem()
        assert k == "foo"
        assert v == "blub"

    def test_clear(self):
        self.view.clear()
        assert len(self.view) == 0
