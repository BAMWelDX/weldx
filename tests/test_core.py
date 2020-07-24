"""Tests of the core package."""

import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.core import MathematicalExpression, TimeSeries

# MathematicalExpression ---------------------------------------------------------------


def test_mathematical_expression_construction():
    """Test the construction of a MathematicalExpression."""
    expr = MathematicalExpression("a*b+c/d-e", parameters={"d": 1, "e": 2})

    assert expr.num_parameters == 2
    assert expr.num_variables == 3

    for variable in ["a", "b", "c"]:
        assert variable in expr.get_variable_names()

    for parameter, value in {"d": 1, "e": 2}.items():
        assert parameter in expr.parameters
        assert expr.parameters[parameter] == value

    # exceptions ------------------------------------------
    # parameter not in expression
    with pytest.raises(ValueError):
        expr = MathematicalExpression("a*b+c/d-e", parameters={"f": 1})
    # invalid parameter type
    with pytest.raises(ValueError):
        expr = MathematicalExpression("a*b+c/d-e", parameters=1)


def test_mathematical_expression_set_parameter():
    """Test the set_parameter function of the mathematical expression."""
    expr = MathematicalExpression("a*b+c/d-e")

    assert expr.num_parameters == 0
    assert expr.num_variables == 5

    for variable in ["a", "b", "c", "d", "e"]:
        assert variable in expr.get_variable_names()

    assert len(expr.parameters) == 0

    # set first parameters
    expr.set_parameter("d", 1)
    expr.set_parameter("e", 2)

    assert expr.num_parameters == 2
    assert expr.num_variables == 3

    for variable in ["a", "b", "c"]:
        assert variable in expr.get_variable_names()

    for parameter, value in {"d": 1, "e": 2}.items():
        assert parameter in expr.parameters
        assert expr.parameters[parameter] == value

    # set another parameter and overwrite others
    expr.set_parameter("a", 5)
    expr.set_parameter("d", 7)
    expr.set_parameter("e", -1)

    assert expr.num_parameters == 3
    assert expr.num_variables == 2

    for variable in ["b", "c"]:
        assert variable in expr.get_variable_names()

    for parameter, value in {"a": 5, "d": 7, "e": -1}.items():
        assert parameter in expr.parameters
        assert expr.parameters[parameter] == value


def test_mathematical_expression_comparison():
    """Test the different comparison functions of the MathematicalExpression."""
    expr_string = "(a + b)**2 + c - d"

    parameters = {"a": 2, "c": 3.5}

    expr = MathematicalExpression(expr_string, parameters)

    # check structurally equal ----------------------------
    expr_equal = MathematicalExpression(expr_string, parameters)
    assert expr == expr_equal
    assert not expr != expr_equal
    assert expr.equals(expr_equal)
    assert expr.equals(expr_equal, check_parameters=False)

    # check mathematical equal expression -----------------
    expr_string_math_equal = "a**2 + 2*a*b + b**2 + c - d"
    expr_math_equal = MathematicalExpression(expr_string_math_equal, parameters)
    assert not expr == expr_math_equal
    assert expr != expr_math_equal
    assert expr.equals(expr_math_equal)
    assert expr.equals(expr_math_equal, check_parameters=False)

    # check totally different expression ------------------
    expr_string_different = "a*b + c*d"
    expr_different = MathematicalExpression(expr_string_different, parameters)
    assert not expr == expr_different
    assert expr != expr_different
    assert not expr.equals(expr_different)
    assert not expr.equals(expr_different, check_parameters=False)

    # check different number of parameters ----------------
    parameters_plus_one = {"a": 2, "c": 3.5, "d": 4}
    expr_plus_one = MathematicalExpression(expr_string, parameters_plus_one)
    assert not expr == expr_plus_one
    assert expr != expr_plus_one
    assert not expr.equals(expr_plus_one)
    assert expr.equals(expr_plus_one, check_parameters=False)

    expr_math_equal_plus_one = MathematicalExpression(
        expr_string_math_equal, parameters_plus_one
    )
    assert not expr == expr_math_equal_plus_one
    assert expr != expr_math_equal_plus_one
    assert not expr.equals(expr_math_equal_plus_one)
    assert expr.equals(expr_math_equal_plus_one, check_parameters=False)

    expr_different_plus_one = MathematicalExpression(
        expr_string_different, parameters_plus_one
    )
    assert not expr == expr_different_plus_one
    assert expr != expr_different_plus_one
    assert not expr.equals(expr_different_plus_one)
    assert not expr.equals(expr_different_plus_one, check_parameters=False)

    # check different parameter values --------------------
    parameters_different = {"a": 2, "c": 3.4}
    expr_different = MathematicalExpression(expr_string, parameters_different)
    assert not expr == expr_different
    assert expr != expr_different
    assert not expr.equals(expr_different)
    assert expr.equals(expr_different, check_parameters=False)

    expr_math_equal_different = MathematicalExpression(
        expr_string_math_equal, parameters_different
    )
    assert not expr == expr_math_equal_different
    assert expr != expr_math_equal_different
    assert not expr.equals(expr_math_equal_different)
    assert expr.equals(expr_math_equal_different, check_parameters=False)

    expr_different_different = MathematicalExpression(
        expr_string_different, parameters_different
    )
    assert not expr == expr_different_different
    assert expr != expr_different_different
    assert not expr.equals(expr_different_different)
    assert not expr.equals(expr_different_different, check_parameters=False)

    # other types -----------------------------------------
    assert not expr == 1
    assert expr != 1
    assert not expr.equals(1)
    assert not expr.equals(1, check_parameters=False)

    assert not expr == "nope"
    assert expr != "nope"
    assert not expr.equals("nope")
    assert not expr.equals("nope", check_parameters=False)


def test_mathematical_function_evaluation():
    """Test the evaluation of the mathematical function."""
    expr = MathematicalExpression("a*b+c/d-e", parameters={"d": 1, "e": 2})

    assert expr.evaluate(a=1, b=2, c=3) == 3

    # exceptions ------------------------------------------
    # input already defined as expression parameter
    with pytest.raises(ValueError):
        expr.evaluate(a=1, b=2, c=3, d=2)
    # not enough values provided
    with pytest.raises(Exception):
        expr.evaluate(a=1, b=2)


# TimeSeries ---------------------------------------------------------------------------


def test_time_series_construction():
    """Test the construction of the TimeSeries class."""
    # single value ----------------------------------------
    value = Q_(1, "m")
    ts_constant = TimeSeries(data=value)

    assert ts_constant.data == value
    assert ts_constant.time is None
    assert ts_constant.interpolation is None
    assert ts_constant.shape == (1,)
    assert value.check(UREG.get_dimensionality(ts_constant.units))

    exp_data_array_constant = DataArray(
        data=Q_([1], "m"),
        dims=["time"],
        coords={"time": pd.TimedeltaIndex([0])},
        attrs={"interpolation": None},
    )
    assert ts_constant.data_array.identical(exp_data_array_constant)

    # discrete values -------------------------------------
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(data=values, time=time, interpolation="step")

    assert np.all(ts_discrete.time == time)
    assert np.all(ts_discrete.data == values)
    assert ts_discrete.interpolation == "step"
    assert ts_discrete.shape == (5,)
    assert values.check(UREG.get_dimensionality(ts_discrete.units))

    exp_data_array_discrete = DataArray(
        data=values,
        dims=["time"],
        coords={"time": time},
        attrs={"interpolation": "step"},
    )
    assert ts_discrete.data_array.identical(exp_data_array_discrete)

    # mathematical expression -----------------------------
    # scalar
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "1/s"), "b": Q_(-2, "")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = TimeSeries(data=expr)

    assert ts_expr.time is None
    assert ts_expr.interpolation is None
    assert ts_expr.shape == (1,)
    assert ts_expr.data_array is None

    assert isinstance(ts_expr.data, MathematicalExpression)
    assert ts_expr.data.num_variables == 1
    assert ts_expr.data.num_parameters == 2
    assert ts_expr.data.get_variable_names()[0] == "t"
    assert parameters["b"].check(UREG.get_dimensionality(ts_expr.units))

    for parameter in parameters:
        assert parameter in ts_expr.data.parameters
        assert parameters[parameter] == ts_expr.data.parameters[parameter]

    # vector
    expr_string_vec = "a * time + b"
    parameters_vec = {"a": Q_([[2, 3, 4]], "m/s"), "b": Q_([[-2, 3, 1]], "m")}
    expr_vec = MathematicalExpression(
        expression=expr_string_vec, parameters=parameters_vec
    )

    ts_expr_vec = TimeSeries(data=expr_vec)

    assert ts_expr_vec.time is None
    assert ts_expr_vec.interpolation is None
    assert ts_expr_vec.shape == (1, 3)
    assert ts_expr_vec.data_array is None

    assert isinstance(ts_expr_vec.data, MathematicalExpression)
    assert ts_expr_vec.data.num_variables == 1
    assert ts_expr_vec.data.num_parameters == 2
    assert ts_expr_vec.data.get_variable_names()[0] == "time"
    assert parameters_vec["b"].check(UREG.get_dimensionality(ts_expr_vec.units))

    for parameter in parameters_vec:
        assert parameter in ts_expr_vec.data.parameters
        assert np.all(
            parameters_vec[parameter] == ts_expr_vec.data.parameters[parameter]
        )

    # exceptions ------------------------------------------
    # invalid interpolation
    with pytest.raises(ValueError):
        TimeSeries(values, time, "super interpolator 2000")
    with pytest.raises(ValueError):
        TimeSeries(values, time, None)

    # too many free variables
    with pytest.raises(Exception):
        expr_2 = MathematicalExpression(expression=expr_string, parameters={})
        TimeSeries(data=expr_2)
    # incompatible parameter units
    with pytest.raises(Exception):
        expr_3 = MathematicalExpression(
            expression=expr_string, parameters={"a": Q_(2, "1/s"), "b": Q_(-2, "m")}
        )
        TimeSeries(data=expr_3)
    # cannot be evaluated with time vectors
    with pytest.raises(Exception):
        expr_4 = MathematicalExpression(
            expression=expr_string,
            parameters={"a": Q_([2, 3, 4], "1/s"), "b": Q_([-2, 3, 1], "")},
        )
        TimeSeries(data=expr_4)


def test_time_series_comparison():
    """Test the comparison functions of the TimeSeries class."""
    value_constant = Q_(1, "m")
    ts_constant = TimeSeries(data=value_constant)

    time_discrete = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values_discrete = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(
        data=values_discrete, time=time_discrete, interpolation="step"
    )

    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "1/s"), "b": Q_(-2, "")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = TimeSeries(data=expr)

    # comparison with duplicate ---------------------------

    assert ts_constant == TimeSeries(value_constant)
    assert ts_discrete == TimeSeries(values_discrete, time_discrete, "step")
    assert ts_expr == TimeSeries(expr)

    assert not ts_constant != TimeSeries(value_constant)
    assert not ts_discrete != TimeSeries(values_discrete, time_discrete, "step")
    assert not ts_expr != TimeSeries(expr)

    # comparison against each other -----------------------

    assert not ts_constant == ts_discrete
    assert not ts_constant == ts_expr
    assert not ts_discrete == ts_expr

    assert ts_constant != ts_discrete
    assert ts_constant != ts_expr
    assert ts_discrete != ts_expr

    # comparison with other type --------------------------

    assert not ts_constant == 1
    assert not ts_discrete == 1
    assert not ts_expr == 1

    assert ts_constant != 1
    assert ts_discrete != 1
    assert ts_expr != 1

    assert not ts_constant == "nope"
    assert not ts_discrete == "nope"
    assert not ts_expr == "nope"

    assert ts_constant != "nope"
    assert ts_discrete != "nope"
    assert ts_expr != "nope"

    # constant value specific mismatches ------------------
    ts_constant_value_wrong = TimeSeries(Q_(1337, "m"))
    assert not ts_constant == ts_constant_value_wrong
    assert ts_constant != ts_constant_value_wrong

    ts_constant_unit_wrong = TimeSeries(Q_(1, "s"))
    assert not ts_constant == ts_constant_unit_wrong
    assert ts_constant != ts_constant_unit_wrong

    ts_constant_unit_prefix_wrong = TimeSeries(Q_(1, "mm"))
    assert not ts_constant == ts_constant_unit_prefix_wrong
    assert ts_constant != ts_constant_unit_prefix_wrong

    # discrete value specific mismatches ------------------
    time_wrong = pd.TimedeltaIndex([0, 1, 2, 3, 5])
    ts_discrete_time_wrong = TimeSeries(values_discrete, time_wrong, "step")
    assert not ts_discrete == ts_discrete_time_wrong
    assert ts_discrete != ts_discrete_time_wrong

    values_discrete_wrong = Q_(np.array([10, 11, 12, 15, 16]), "mm")
    ts_discrete_values_wrong = TimeSeries(values_discrete_wrong, time_discrete, "step")
    assert not ts_discrete == ts_discrete_values_wrong
    assert ts_discrete != ts_discrete_values_wrong

    values_unit_wrong = Q_(np.array([10, 11, 12, 14, 16]), "s")
    ts_discrete_unit_wrong = TimeSeries(values_unit_wrong, time_discrete, "step")
    assert not ts_discrete == ts_discrete_unit_wrong
    assert ts_discrete != ts_discrete_unit_wrong

    values_unit_prefix_wrong = Q_(np.array([10, 11, 12, 14, 16]), "m")
    ts_discrete_unit_prefix_wrong = TimeSeries(
        values_unit_prefix_wrong, time_discrete, "step"
    )
    assert not ts_discrete == ts_discrete_unit_prefix_wrong
    assert ts_discrete != ts_discrete_unit_prefix_wrong

    ts_discrete_interp_wrong = TimeSeries(values_discrete, time_discrete, "linear")
    assert not ts_discrete == ts_discrete_interp_wrong
    assert ts_discrete != ts_discrete_interp_wrong

    # expression specific mismatches ----------------------

    expr_wrong_exp = MathematicalExpression("a*t+ 2*b", parameters)
    ts_expr_wrong_expr = TimeSeries(expr_wrong_exp)
    assert not ts_expr == ts_expr_wrong_expr
    assert ts_expr != ts_expr_wrong_expr

    parameters_wrong_values = {"a": Q_(2, "1/s"), "b": Q_(-1, "")}
    expr_wrong_param_values = MathematicalExpression(
        expr_string, parameters_wrong_values
    )
    ts_expr_wrong_param_values = TimeSeries(expr_wrong_param_values)
    assert not ts_expr == ts_expr_wrong_param_values
    assert ts_expr != ts_expr_wrong_param_values

    parameters_wrong_unit = {"a": Q_(2, "m/s"), "b": Q_(-2, "m")}
    expr_wrong_param_unit = MathematicalExpression(expr_string, parameters_wrong_unit)
    ts_expr_wrong_param_unit = TimeSeries(expr_wrong_param_unit)
    assert not ts_expr == ts_expr_wrong_param_unit
    assert ts_expr != ts_expr_wrong_param_unit

    parameters_wrong_unit_prefix = {"a": Q_(2, "1/ms"), "b": Q_(-2, "")}
    expr_wrong_param_unit_prefix = MathematicalExpression(
        expr_string, parameters_wrong_unit_prefix
    )
    ts_expr_wrong_param_unit_prefix = TimeSeries(expr_wrong_param_unit_prefix)
    assert not ts_expr == ts_expr_wrong_param_unit_prefix
    assert ts_expr != ts_expr_wrong_param_unit_prefix


def test_time_series_interp_time_constant():
    """Test the TimeSeries.inter_time method for constants as data."""
    value = Q_(1, "m")
    ts_constant = TimeSeries(data=value)

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([2], "s")
    value_interp_single = ts_constant.interp_time(time_delta_single)

    assert value_interp_single == value

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([0, 2, 5], "s")
    value_interp_multi = ts_constant.interp_time(time_delta_multi)

    assert len(value_interp_multi) == 3
    for value_interp in value_interp_multi:
        assert value_interp == value


def test_time_series_interp_time_discrete_step():
    """Test the inter_time method for discrete data and step interpolation."""
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(data=values, time=time, interpolation="step")

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([1.5], "s")
    value_interp_single = ts_discrete.interp_time(time_delta_single)

    assert np.isclose(value_interp_single.data.magnitude, 11)
    assert value_interp_single.data.check(values.dimensionality)

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
    value_interp_multi = ts_discrete.interp_time(time_delta_multi)

    assert np.all(
        np.isclose(value_interp_multi.data.magnitude, [10, 10, 11, 11, 12, 14, 16, 16])
    )
    assert value_interp_multi.data.check(values.dimensionality)


def test_time_series_interp_time_discrete_linear():
    """Test the inter_time method for discrete data and linear interpolation."""
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(data=values, time=time, interpolation="linear")

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([1.5], "s")
    value_interp_single = ts_discrete.interp_time(time_delta_single)

    assert np.isclose(value_interp_single.data.magnitude, 11.5)
    assert value_interp_single.data.check(values.dimensionality)

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([-3, 2.5, 3, 4, 7], "s")
    value_interp_multi = ts_discrete.interp_time(time_delta_multi)

    assert np.all(np.isclose(value_interp_multi.data.magnitude, [10, 13, 14, 16, 16]))
    assert value_interp_multi.data.check(values.dimensionality)


def test_time_series_interp_time_expression():
    """Test the TimeSeries.inter_time method for mathematical expressions as data."""
    # scalar ----------------------------------------------
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "meter/second"), "b": Q_(-2, "meter")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = TimeSeries(data=expr)

    # single timedelta
    time_single = Q_(1, "second")
    value_interp_single = ts_expr.interp_time(time_single)

    assert value_interp_single.data == Q_(0, "meter")

    # multiple time deltas
    time_multi = Q_([0, 1, 2, 10], "second")
    value_interp_multi = ts_expr.interp_time(time_multi)

    assert len(value_interp_multi) == 4

    for i in range(4):
        assert (
            value_interp_multi[i].data
            == parameters["a"] * time_multi[i] + parameters["b"]
        )

    # vector -----------------------------------------------
    expr_string_vec = "a*t+b"
    parameters_vec = {"a": Q_([[2, 3, 4]], "1/s"), "b": Q_([[-2, 3, 1]], "")}
    expr_vec = MathematicalExpression(
        expression=expr_string_vec, parameters=parameters_vec
    )

    ts_expr_vec = TimeSeries(data=expr_vec)

    # single time delta
    value_interp_vec_single = ts_expr_vec.interp_time(time_single)

    assert np.all(np.isclose(value_interp_vec_single.data.magnitude, [0, 6, 5]))

    # multiple time deltas
    value_interp_vec_multi = ts_expr_vec.interp_time(time_multi)

    assert value_interp_vec_multi.shape == (4, 3)

    for i in range(4):
        assert (
            value_interp_multi[i].data
            == parameters["a"] * time_multi[i] + parameters["b"]
        )

    # exceptions ------------------------------------------
    with pytest.raises(ValueError):
        ts_expr.interp_time(Q_(2, "s/m"))
