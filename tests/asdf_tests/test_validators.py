import numpy as np
import pandas as pd
import pytest
from asdf import ValidationError

from weldx import Q_
from weldx.asdf.tags.weldx.debug.test_property_tag import PropertyTagTestClass
from weldx.asdf.tags.weldx.debug.test_shape_validator import ShapeValidatorTestClass
from weldx.asdf.tags.weldx.debug.test_unit_validator import UnitValidatorTestClass
from weldx.asdf.validators import _custom_shape_validator

from .utility import _write_read_buffer


@pytest.mark.parametrize(
    "test_input",
    [
        PropertyTagTestClass(),
        pytest.param(
            PropertyTagTestClass(prop3=pd.Timedelta(2, "s")),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            PropertyTagTestClass(prop3="STRING"),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
    ],
)
def test_property_tag_validator(test_input):
    """Test custom ASDF shape validators."""
    _write_read_buffer({"root_node": test_input})


def test_shape_validator_syntax():
    """Test handling of custom shape validation syntax in Python."""

    def val(list_test, list_expected):
        """Add shape key to lists."""
        try:
            _custom_shape_validator({"shape": list_test}, list_expected)
            return True
        except ValidationError:
            return False

    # correct evaluation
    assert val([3], [3])
    assert val([2, 4, 5], [2, 4, 5])
    assert val([1, 2, 3], ["..."])
    assert val([1, 2], [1, 2, "..."])
    assert val([1, 2], ["...", 1, 2])
    assert val([1, 2, 3], [1, 2, None])
    assert val([1, 2, 3], [None, 2, 3])
    assert val([1], [1, "..."])
    assert val([1, 2, 3, 4, 5], [1, "..."])
    assert val([1, 2, 3, 4, 5], ["...", 4, 5])
    assert val([1, 2], [1, 2, "(3)"])
    assert val([2, 3], ["(1)", 2, 3])
    assert val([1, 2, 3], ["(1)", 2, 3])
    assert val([2, 3], ["(1~3)", 2, 3])
    assert val([2, 2, 3], ["(1~3)", 2, 3])
    assert val([1, 2, 3], [1, "1~3", 3])
    assert val([1, 2, 3], [1, "1~", 3])
    assert val([1, 2, 3], [1, "~3", 3])
    assert val([1, 2, 3], [1, "~", 3])
    assert val([1, 200, 3], [1, "~", 3])
    assert val([1, 2, 3], [1, 2, "(~)"])
    assert val([1, 2, 300], [1, 2, "(~)"])
    # assert val([1, 2, 3], [1, "(n)", "..."])  # should this be allowed?
    _custom_shape_validator(1.0, [1])

    # shape mismatch
    assert not val([2, 2, 3], [1, "..."])
    assert not val([2, 2, 3], ["...", 1])
    assert not val([1], [1, 2])
    assert not val([1, 2], [1])
    assert not val([1, 2], [3, 2])
    assert not val([1], [1, "~"])
    assert not val([1], ["~", 1])
    assert not val([1, 2, 3], [1, 2, "(4)"])
    assert not val([1, 2, 3], ["(2)", 2, 3])
    assert not val([1, 2], [1, "4~8"])
    assert not val([1, 9], [1, "4~8"])
    assert not val([1, 2], [1, "(4~8)"])
    assert not val([1, 9], [1, "(4~8)"])

    # syntax errors, these should throw a ValueError
    with pytest.raises(ValueError):
        val([1, 2], [1, "~", "(...)"])  # value error?
    with pytest.raises(ValueError):
        val([1, 2], [1, "(2)", 3])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "((3))"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "3)"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "*3"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "(3"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "(3)3"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "2(3)"])
    with pytest.raises(ValueError):
        val([1, 2], [1, "...", 2])  # should this be allowed? syntax/value error?
    with pytest.raises(ValueError):
        val([1, 2], ["(1)", "..."])
    with pytest.raises(ValueError):
        val([1, 2], [1, "4~1"])
    # no negative shape numbers allowed in syntax
    with pytest.raises(ValueError):
        val([-1, -2], [-1, -2])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "(-3)"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "(-3~-1)"])
    with pytest.raises(ValueError):
        val([1, 2], [1, 2, "(-3~1)"])
    with pytest.raises(ValueError):
        val([1, 2, 1], ["(-3~1)", 2, 1])
    with pytest.raises(ValidationError):  # test single value
        _custom_shape_validator(1.0, [2])


@pytest.mark.parametrize(
    "test_input",
    [
        ShapeValidatorTestClass(
            prop1=np.ones((1, 2, 3)),
            prop2=np.ones((3, 2, 1)),
            prop3=np.ones((2, 4, 6, 8, 10)),
            prop4=np.ones((1, 3, 5, 7, 9)),
            prop5=3.141,
            nested_prop={
                "p1": np.ones((10, 8, 6, 4, 2)),
                "p2": np.ones((9, 7, 5, 3, 1)),
            },
        ),
        pytest.param(
            ShapeValidatorTestClass(
                prop1=np.ones((1, 2, 3)),
                prop2=np.ones((3, 2, 1)),
                prop3=np.ones((2, 4, 6, 8, 10)),
                prop4=np.ones((2, 3, 5, 7, 9)),  # mismatch a with prop5
                prop5=3.141,
                nested_prop={
                    "p1": np.ones((10, 8, 6, 4, 2)),
                    "p2": np.ones((9, 7, 5, 3, 1)),
                },
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            ShapeValidatorTestClass(
                prop1=np.ones((1, 2, 3)),
                prop2=np.ones((5, 2, 1)),  # mismatch n with prop1
                prop3=np.ones((2, 4, 6, 8, 10)),
                prop4=np.ones((1, 3, 5, 7, 9)),
                prop5=3.141,
                nested_prop={
                    "p1": np.ones((10, 8, 6, 4, 2)),
                    "p2": np.ones((9, 7, 5, 3, 1)),
                },
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            ShapeValidatorTestClass(
                prop1=np.ones((1, 2, 3)),
                prop2=np.ones((3, 2, 1)),
                prop3=np.ones((2, 4, 6, 8, 10)),
                prop4=np.ones((1, 3, 5, 7, 9)),
                prop5=3.141,
                nested_prop={"p1": np.ones((10, 8, 6, 4, 2))},  # missing p2
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
    ],
)
def test_shape_validator(test_input):
    _write_read_buffer({"root": test_input})


@pytest.mark.parametrize(
    "test",
    [
        UnitValidatorTestClass(
            length_prop=Q_(1, "inch"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "m"},
        ),
        pytest.param(
            UnitValidatorTestClass(
                length_prop=Q_(1, "s"),  # wrong unit
                velocity_prop=Q_(2, "km / s"),
                current_prop=Q_(np.eye(2, 2), "mA"),
                nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
                simple_prop={"value": float(3), "unit": "m"},
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            UnitValidatorTestClass(
                length_prop=Q_(1, "s"),
                velocity_prop=Q_(2, "liter"),  # wrong unit
                current_prop=Q_(np.eye(2, 2), "mA"),
                nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
                simple_prop={"value": float(3), "unit": "m"},
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            UnitValidatorTestClass(
                length_prop=Q_(1, "inch"),
                velocity_prop=Q_(2, "km / s"),
                current_prop=Q_(np.eye(2, 2), "V"),  # wrong unit
                nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
                simple_prop={"value": float(3), "unit": "m"},
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            UnitValidatorTestClass(
                length_prop=Q_(1, "m"),
                velocity_prop=Q_(2, "km / s"),
                current_prop=Q_(np.eye(2, 2), "mA"),
                nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "V")),  # wrong unit
                simple_prop={"value": float(3), "unit": "m"},
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
        pytest.param(
            UnitValidatorTestClass(
                length_prop=Q_(1, "m"),
                velocity_prop=Q_(2, "km / s"),
                current_prop=Q_(np.eye(2, 2), "mA"),
                nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
                simple_prop={"value": float(3), "unit": "s"},  # wrong unit
            ),
            marks=pytest.mark.xfail(raises=ValidationError),
        ),
    ],
)
def test_unit_validator(test):
    data = _write_read_buffer({"root_node": test})
    test_read = data["root_node"]
    assert isinstance(data, dict)
    assert test_read.length_prop == test.length_prop
    assert test_read.velocity_prop == test.velocity_prop
    assert np.all(test_read.current_prop == test.current_prop)
    assert np.all(test_read.nested_prop["q1"] == test.nested_prop["q1"])
    assert test_read.nested_prop["q2"] == test.nested_prop["q2"]
    assert test_read.simple_prop == test.simple_prop
