"""tests for asdf utility functions."""
from dataclasses import dataclass
from typing import List

import pytest

from weldx import WeldxFile
from weldx.asdf.util import dataclass_serialization_class, get_yaml_header


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


def _to_tree_mod(tree):
    tree["a"] += ["d"]
    return tree


def _from_tree_mod(tree):
    tree["a"] += ["e"]
    return tree


@pytest.mark.parametrize(
    "val_a, exp_val_a_tree, exp_val_a_dc, to_tree_mod, from_tree_mod,"
    "sort_string_lists",
    [
        (["c", "b", "a"], ["a", "b", "c"], ["a", "b", "c"], None, None, True),
        (["c", "b", "a"], ["c", "b", "a"], ["c", "b", "a"], None, None, False),
        # not a pure string list -> no sorting
        (["c", 1, "a"], ["c", 1, "a"], ["c", 1, "a"], None, None, True),
        (["c", 1, "a"], ["c", 1, "a"], ["c", 1, "a"], None, None, False),
        # check to_tree_mod is called
        (["c", "b"], ["b", "c", "d"], ["b", "c", "d"], _to_tree_mod, None, True),
        (["c", "b"], ["c", "b", "d"], ["c", "b", "d"], _to_tree_mod, None, False),
        # check_from_tree_mod is called
        (["c"], ["c", "d"], ["c", "d", "e"], _to_tree_mod, _from_tree_mod, False),
        (["c"], ["c"], ["c", "e"], None, _from_tree_mod, False),
    ],
)
def test_dataclass_serialization_class(
    val_a, exp_val_a_tree, exp_val_a_dc, to_tree_mod, from_tree_mod, sort_string_lists
):
    """Test the `dataclass_serialization_class` function.

    The test defines a dataclass and its corresponding serialization class using
    `dataclass_serialization_class`. It first calls to_tree to get tree from the
    generated serialization class. Afterwards the tree is used with the `from_tree`
    method to construct a new dataclass instance. The results of the function calls are
    checked against the expected values.

    Parameters
    ----------
    val_a :
        Initial value of the dataclasses' variable a
    exp_val_a_tree :
        Expected value of the variable a in the tree after `to_tree` was run
    exp_val_a_dc :
        Expected value of the variable a of the reconstructed dataclass
    to_tree_mod :
        The value passed as corresponding function parameter
    from_tree_mod :
        The value passed as corresponding function parameter
    sort_string_lists
        The value passed as corresponding function parameter


    """

    @dataclass
    class _DataClass:
        a: List[str]
        b: int = 1

    dc = _DataClass(a=val_a, b=2)

    dataclass_asdf = dataclass_serialization_class(
        class_type=_DataClass,
        class_name="Test",
        version="1.0.0",
        sort_string_lists=sort_string_lists,
        to_tree_mod=to_tree_mod,
        from_tree_mod=from_tree_mod,
    )
    tree = dataclass_asdf.to_tree(dc, None)

    assert tree["b"] == 2
    assert tree["a"] == exp_val_a_tree

    dc_restored = dataclass_asdf.from_tree(tree, None)

    assert dc_restored.b == 2
    assert dc_restored.a == exp_val_a_dc
