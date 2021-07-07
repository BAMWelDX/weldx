"""Test custom weldx ASDF validator functions."""
import numpy as np
import pandas as pd
import pytest
from asdf import ValidationError

from weldx import Q_, TimeSeries
from weldx.asdf.extension import WxSyntaxError
from weldx.asdf.tags.weldx.debug.test_property_tag import PropertyTagTestClass
from weldx.asdf.tags.weldx.debug.test_shape_validator import ShapeValidatorTestClass
from weldx.asdf.tags.weldx.debug.test_unit_validator import UnitValidatorTestClass
from weldx.asdf.util import write_buffer, write_read_buffer
from weldx.asdf.validators import _compare_tag_version, _custom_shape_validator
from weldx.util import compare_nested


@pytest.mark.parametrize(
    "instance_tag,tagname,result",
    [
        (None, "tag:debug.com/object-*", True),
        ("tag:debug.com/object-1.2.3", "tag:debug.com/object-*", True),
        ("http://debug.com/object-1.2.3", "http://debug.com/object-*", True),
        ("http://debug.com/object-1.2.3", "http://debug.com/object-1.2.3", True),
        ("http://debug.com/object-1.2.3", "http://debug.com/object-1.2", True),
        ("http://debug.com/object-1.2.3", "http://debug.com/object-1", True),
        ("http://debug.com/object-1.2.3", "http://debug.com/object-2", False),
        ("http://debug.com/object-2.0.0", "http://debug.com/object-1", False),
        ("http://debug.com/object-2.0.0", "http://debug.com/object-2.1", False),
        ("http://debug.com/object-2.0.0", "http://debug.com/other-2.0.0", False),
        ("http://debug.com/object-2.0.0", "http://other.com/object-2.0.0", False),
        ("http://debug.com/object-1.2.3", "http://other.com/object-1.2.3", False),
    ],
)
def test_wx_tag_syntax(instance_tag, tagname, result):
    """Test ASDF tag version syntax resolving."""
    assert _compare_tag_version(instance_tag, tagname) == result


@pytest.mark.parametrize(
    "instance_tag,tagname,err",
    [
        ("tag:debug.com/object-1.2.3", "tag:debug.com/object", WxSyntaxError),
        ("tag:debug.com/object-1.2.3", "tag:debug.com/object-", WxSyntaxError),
        ("tag:debug.com/object-1.2.3", "tag:debug.com/object-**", WxSyntaxError),
    ],
)
def test_wx_tag_syntax_exceptions(instance_tag, tagname, err):
    """Test custom ASDF shape validators."""
    with pytest.raises(err):
        _compare_tag_version(instance_tag, tagname)


@pytest.mark.parametrize(
    "test_input",
    [PropertyTagTestClass()],
)
def test_property_tag_validator(test_input):
    """Test custom ASDF shape validators."""
    write_read_buffer({"root_node": test_input})


@pytest.mark.parametrize(
    "test_input,err",
    [
        (PropertyTagTestClass(prop3=pd.Timedelta(2, "s")), ValidationError),
        (PropertyTagTestClass(prop3="STRING"), ValidationError),
    ],
)
def test_property_tag_validator_exceptions(test_input, err):
    """Test custom ASDF shape validators."""
    with pytest.raises(err):
        write_read_buffer({"root_node": test_input})


def _val(list_test, list_expected):
    """Add shape key to lists."""
    if isinstance(list_test, list):
        res = _custom_shape_validator({"shape": list_test}, list_expected)
        return isinstance(res, dict)
    return isinstance(_custom_shape_validator(list_test, list_expected), dict)


@pytest.mark.parametrize(
    "shape, exp",
    [
        ([3], [3]),
        ([2, 4, 5], [2, 4, 5]),
        ([1, 2, 3], ["..."]),
        ([1, 2], [1, 2, "..."]),
        ([1, 2], ["...", 1, 2]),
        ([1, 2, 3], [1, 2, None]),
        ([1, 2, 3], [None, 2, 3]),
        ([1], [1, "..."]),
        ([1, 2, 3, 4, 5], [1, "..."]),
        ([1, 2, 3, 4, 5], ["...", 4, 5]),
        ([1, 2], [1, 2, "(3)"]),
        ([1, 2], [1, 2, "(n)"]),
        ([1, 2], [1, 2, "(2)", "(3)"]),
        ([2, 3], ["(1)", 2, 3]),
        ([1, 2, 3], ["(1)", 2, 3]),
        ([2, 3], ["(1~3)", 2, 3]),
        ([2, 2, 3], ["(1~3)", 2, 3]),
        ([1, 2, 3], [1, "1~3", 3]),
        ([1, 2, 3], [1, "1~", 3]),
        ([1, 2, 3], [1, "~3", 3]),
        ([1, 2, 3], [1, "~", 3]),
        ([1, 200, 3], [1, "~", 3]),
        ([1, 2, 3], [1, 2, "(~)"]),
        ([1, 2, 300], [1, 2, "(~)"]),
        ([1, 2, 3], [1, "(n)", "..."]),
        (1.0, [1]),
    ],
)
def test_shape_validator_syntax2(shape, exp):
    assert _val(shape, exp)


@pytest.mark.parametrize(
    "shape, exp, err",
    [
        ([2, 2, 3], [1, "..."], ValidationError),
        ([2, 2, 3], ["...", 1], ValidationError),
        ([1], [1, 2], ValidationError),
        ([1, 2], [1], ValidationError),
        ([1, 2], [3, 2], ValidationError),
        ([1], [1, "~"], ValidationError),
        ([1], ["~", 1], ValidationError),
        ([1, 2, 3], [1, 2, "(4)"], ValidationError),
        ([1, 2, 3], ["(2)", 2, 3], ValidationError),
        ([1, 2], [1, "4~8"], ValidationError),
        ([1, 9], [1, "4~8"], ValidationError),
        ([1, 2], [1, "(4~8)"], ValidationError),
        ([1, 9], [1, "(4~8)"], ValidationError),
        (1.0, [2], ValidationError),
        ([1, 2, 3, 4], [1, 2, "n", "n"], ValidationError),
        ([1, 2], [1, "~", "(...)"], WxSyntaxError),
        ([1, 2], [1, "(2)", 3], WxSyntaxError),
        ([1, 2], [1, 2, "((3))"], WxSyntaxError),
        ([1, 2], [1, 2, "3)"], WxSyntaxError),
        ([1, 2], [1, 2, "*3"], WxSyntaxError),
        ([1, 2], [1, 2, "(3"], WxSyntaxError),
        ([1, 2], [1, 2, "(3)3"], WxSyntaxError),
        ([1, 2], [1, 2, "2(3)"], WxSyntaxError),
        ([1, 2], [1, "...", 2], WxSyntaxError),
        ([1, 2], ["(1)", "..."], WxSyntaxError),
        ([1, 2], [1, "4~1"], WxSyntaxError),
        ([-1, -2], [-1, -2], WxSyntaxError),
        ([-1, 2], [1, 2], WxSyntaxError),
        ([1, 2], [-1, 2], WxSyntaxError),
        ([1, 2], [1, 2, "(-3)"], WxSyntaxError),
        ([1, 2], [1, 2, "(-3~-1)"], WxSyntaxError),
        ([1, 2], [1, 2, "(-3~1)"], WxSyntaxError),
        ([1, 2, 1], ["(-3~1)", 2, 1], WxSyntaxError),
        ([1, 2], [1, "(9~m)"], WxSyntaxError),
        ([1, 2], [1, "(n~9)"], WxSyntaxError),
        ([1, 2], [1, "(n~m)"], WxSyntaxError),
        ([1, 2], [1, "(1~3~5)"], WxSyntaxError),
        ("a string", [1, "(1~3~5)"], ValidationError),
        ([1, 2], "a string", WxSyntaxError),
    ],
)
def test_shape_validation_error_exception(shape, exp, err):
    with pytest.raises(err):
        assert _val(shape, exp)


@pytest.mark.parametrize(
    "test_input",
    [
        ShapeValidatorTestClass(),
        ShapeValidatorTestClass(time_prop=pd.date_range("2020", freq="D", periods=9)),
        ShapeValidatorTestClass(
            optional_prop=np.ones((1, 2, 3)),
        ),
        ShapeValidatorTestClass(
            nested_prop={
                "p1": np.ones((10, 8, 6, 4, 2)),
                "p2": np.ones((9, 7, 5, 3, 1)),
                "p3": np.ones((1, 2, 3)),
            }
        ),
    ],
)
def test_shape_validator(test_input):
    result = write_read_buffer(
        {"root": test_input},
    )["root"]
    assert compare_nested(test_input.__dict__, result.__dict__)
    assert compare_nested(result.__dict__, test_input.__dict__)


@pytest.mark.parametrize(
    "test_input",
    [
        ShapeValidatorTestClass(
            prop4=np.ones((2, 3, 5, 7, 9)),  # mismatch a with prop5
        ),
        ShapeValidatorTestClass(
            prop2=np.ones((5, 2, 1)),
        ),  # mismatch n with prop1
        ShapeValidatorTestClass(
            nested_prop={"p1": np.ones((10, 8, 6, 4, 2))},  # missing p2
        ),
        ShapeValidatorTestClass(
            optional_prop=np.ones((3, 2, 9)),
        ),  # wrong optional
        ShapeValidatorTestClass(time_prop=pd.date_range("2020", freq="D", periods=3)),
        ShapeValidatorTestClass(
            quantity=Q_([0, 3], "s"),  # mismatch shape [1]
        ),
        ShapeValidatorTestClass(
            timeseries=TimeSeries(
                Q_([0, 3], "m"), Q_([0, 1], "s")
            )  # mismatch shape [1]
        ),
    ],
)
def test_shape_validator_exceptions(test_input):
    with pytest.raises(ValidationError):
        write_read_buffer({"root": test_input})


@pytest.mark.parametrize(
    "test",
    [
        UnitValidatorTestClass(),
        UnitValidatorTestClass(length_prop=Q_(1, "inch")),
    ],
)
def test_unit_validator(test):
    data = write_read_buffer({"root_node": test})
    test_read = data["root_node"]
    assert isinstance(data, dict)
    assert test_read.length_prop == test.length_prop
    assert test_read.velocity_prop == test.velocity_prop
    assert np.all(test_read.current_prop == test.current_prop)
    assert np.all(test_read.nested_prop["q1"] == test.nested_prop["q1"])
    assert test_read.nested_prop["q2"] == test.nested_prop["q2"]
    assert test_read.simple_prop == test.simple_prop


@pytest.mark.parametrize(
    "test",
    [
        UnitValidatorTestClass(
            length_prop=Q_(1, "s"),  # wrong unit
        ),
        UnitValidatorTestClass(
            velocity_prop=Q_(2, "liter"),  # wrong unit
        ),
        UnitValidatorTestClass(
            current_prop=Q_(np.eye(2, 2), "V"),  # wrong unit
        ),
        UnitValidatorTestClass(
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "V")),  # wrong unit
        ),
        UnitValidatorTestClass(
            simple_prop={"value": float(3), "unit": "s"},  # wrong unit
        ),
    ],
)
def test_unit_validator_exception(test):
    with pytest.raises(ValidationError):
        write_buffer({"root_node": test})
