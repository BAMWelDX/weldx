"""Tests of the core package."""


import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.core import MathematicalExpression, TimeSeries

# Todo: Move this to conftest.py?
def get_test_name(param):
    """Get the test name from the parameter list of a parametrized test."""
    if isinstance(param, str) and param[0] == "#":
        return param[1:]
    return ""


# --------------------------------------------------------------------------------------
# MathematicalExpression
# --------------------------------------------------------------------------------------


class TestMathematicalExpression:
    """Tests the mathematical expression class."""

    # Fixtures and variables -----------------------------------------------------------

    ME = MathematicalExpression
    # unfortunately, fixtures can not be used in a parametrize section
    expr_def = "(a + b)**2 + c - d"
    params_def = {"a": 2, "c": 3.5}

    @pytest.fixture()
    def ma_def(self) -> MathematicalExpression:
        """Get a default instance for tests."""
        return MathematicalExpression(
            TestMathematicalExpression.expr_def, TestMathematicalExpression.params_def,
        )

    # Helper functions -----------------------------------------------------------------

    @staticmethod
    def _check_params_and_vars(expression, exp_params, exp_vars):
        """Check parameters and variables of an MathematicalExpression."""
        assert expression.num_parameters == len(exp_params)
        assert len(expression.parameters) == len(exp_params)
        for parameter, value in exp_params.items():
            assert parameter in expression.parameters
            assert expression.parameters[parameter] == value

        assert expression.num_variables == len(exp_vars)
        for variable in exp_vars:
            assert variable in expression.get_variable_names()

    # Tests ----------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "expression, parameters,  exp_vars",
        [
            ("a*b + c/d - e", {"d": 1, "e": 2}, ["a", "b", "c"]),
            ("a*b + c/d - e", {}, ["a", "b", "c", "d", "e"]),
            ("a**2 + b - c", {"a": 1, "c": 2}, ["b"]),
        ],
    )
    def test_construction(self, expression, parameters, exp_vars):
        """Test the construction"""
        expr = MathematicalExpression(expression=expression, parameters=parameters)

        assert expr.num_variables == len(exp_vars)
        for variable in exp_vars:
            assert variable in expr.get_variable_names()

        assert expr.num_parameters == len(parameters)
        for parameter, value in parameters.items():
            assert parameter in expr.parameters
            assert expr.parameters[parameter] == value

    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "expression, parameters, exception_type, name",
        [
            ("a*b + c/d - e", {"f": 1}, ValueError, "# parameter not in expression"),
            ("a*b + c/d - e", 1, ValueError, "# invalid parameter type"),
            ("a + $b#!==3", {"a": 1}, Exception, "# invalid expression"),
        ],
        ids=get_test_name,
    )
    def test_construction_exceptions(
        self, expression, parameters, exception_type, name
    ):
        """Test the exceptions of the '__init__' method."""
        with pytest.raises(exception_type):
            MathematicalExpression(expression=expression, parameters=parameters)

    # -----------------------------------------------------

    def test_set_parameter(self):
        """Test the set_parameter function of the mathematical expression."""
        expr = MathematicalExpression("a*b + c/d - e")

        # check initial configuration
        self._check_params_and_vars(expr, {}, ["a", "b", "c", "d", "e"])

        # set first parameters
        expr.set_parameter("d", 1)
        expr.set_parameter("e", 2)

        self._check_params_and_vars(expr, {"d": 1, "e": 2}, ["a", "b", "c"])

        # set another parameter and overwrite others
        expr.set_parameter("a", 5)
        expr.set_parameter("d", 7)
        expr.set_parameter("e", -1)

        self._check_params_and_vars(expr, {"a": 5, "d": 7, "e": -1}, ["b", "c"])

    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "name, value, exception_type, test_name",
        [
            ("k", 1, ValueError, "# parameter not in expression"),
            (33, 1, TypeError, "# wrong type as name #1"),
            ({"a": 1}, 1, TypeError, "# wrong type as name #2"),
        ],
        ids=get_test_name,
    )
    def test_set_parameter_exceptions(
        self, ma_def, name, value, exception_type, test_name
    ):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ma_def.set_parameter(name, value)

    # -----------------------------------------------------

    expr_mat_identical = "a**2 + 2*a*b + b**2 + c - d"
    expr_different = "a*b + c*d"
    params_too_many = {"a": 2, "c": 3.5, "d": 4}
    params_wrong_value = {"a": 2, "c": 1.5}

    @pytest.mark.parametrize(
        "other, equal, equal_no_params, mat_equal, mat_equal_no_params",
        [
            (ME(expr_def, params_def), True, True, True, True),
            (ME(expr_mat_identical, params_def), False, False, True, True),
            (ME(expr_different, params_def), False, False, False, False),
            (ME(expr_def, params_too_many), False, True, False, True),
            (ME(expr_mat_identical, params_too_many), False, False, False, True),
            (ME(expr_different, params_too_many), False, False, False, False),
            (ME(expr_def, params_wrong_value), False, True, False, True),
            (ME(expr_mat_identical, params_wrong_value), False, False, False, True),
            (ME(expr_different, params_wrong_value), False, False, False, False),
            (1, False, False, False, False),
            ("I am not a MathematicalExpression", False, False, False, False),
        ],
    )
    def test_comparison(
        self, ma_def, other, equal, equal_no_params, mat_equal, mat_equal_no_params,
    ):
        """Test if another object is equal to the default instance."""
        assert (ma_def == other) is equal
        assert (ma_def != other) is not equal
        assert ma_def.equals(other, False, True) is equal_no_params
        assert ma_def.equals(other) is mat_equal
        assert ma_def.equals(other, False, False) is mat_equal_no_params

    # -----------------------------------------------------

    # TODO: Add tests for quantities
    @pytest.mark.parametrize(
        "expression, parameters, variables, exp_result",
        [
            ("a*b + c/d - e", {"d": 1, "e": 2}, {"a": 1, "b": 2, "c": 3}, 3),
            ("(a + b)**2 + c - d", {"a": 3, "d": 2}, {"b": 2, "c": 4}, 27),
            ("a + b", {"a": np.array([1, 2])}, {"b": np.array([2, 4])}, [3, 6],),
        ],
    )
    def test_evaluation(self, expression, parameters, variables, exp_result):
        """Test the evaluation of the mathematical function."""
        expr = MathematicalExpression(expression=expression, parameters=parameters)

        assert np.all(expr.evaluate(**variables) == exp_result)

    # -----------------------------------------------------

    @pytest.mark.parametrize(
        "variables, exception_type, test_name",
        [
            ({"b": 1, "c": 2, "d": 3}, ValueError, "# input is expression parameter"),
            ({"b": 1}, Exception, "# not enough values provided"),
        ],
        ids=get_test_name,
    )
    def test_evaluate_exceptions(self, ma_def, variables, exception_type, test_name):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ma_def.evaluate(**variables)


# --------------------------------------------------------------------------------------
# TimeSeries
# --------------------------------------------------------------------------------------


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

    # discrete values - pint time quantity ----------------
    time_pint = Q_([0, 1, 2, 3, 4], "s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete_pint = TimeSeries(data=values, time=time_pint, interpolation="step")

    assert np.all(ts_discrete_pint.time == time)
    assert np.all(ts_discrete_pint.data == values)
    assert ts_discrete_pint.interpolation == "step"
    assert ts_discrete_pint.shape == (5,)
    assert values.check(UREG.get_dimensionality(ts_discrete_pint.units))

    assert ts_discrete_pint.data_array.identical(exp_data_array_discrete)

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
    # invalid time component
    with pytest.raises(ValueError):
        TimeSeries(values, values.magnitude)

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
    time_delta_single_q = Q_(2, "s")
    value_interp_single = ts_constant.interp_time(time_delta_single)
    value_interp_single_q = ts_constant.interp_time(time_delta_single_q)

    assert value_interp_single == value
    assert value_interp_single_q == value

    assert value_interp_single.time == time_delta_single
    assert value_interp_single_q.time == time_delta_single

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([0, 2, 5], "s")
    time_delta_multi_q = Q_([0, 2, 5], "s")
    value_interp_multi = ts_constant.interp_time(time_delta_multi)
    value_interp_multi_q = ts_constant.interp_time(time_delta_multi_q)

    assert len(value_interp_multi) == 3
    assert len(value_interp_multi_q) == 3

    assert np.all(value_interp_multi.time == time_delta_multi)
    assert np.all(value_interp_multi_q.time == time_delta_multi)

    for value_interp in value_interp_multi:
        assert value_interp == value
    for value_interp in value_interp_multi_q:
        assert value_interp == value

    # exceptions ------------------------------------------
    # wrong type
    with pytest.raises(ValueError):
        ts_constant.interp_time(pd.DatetimeIndex(["2010-10-10"]))
    with pytest.raises(ValueError):
        ts_constant.interp_time("str")
    with pytest.raises(ValueError):
        ts_constant.interp_time([1, 2, 3])


def test_time_series_interp_time_discrete_step():
    """Test the inter_time method for discrete data and step interpolation."""
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(data=values, time=time, interpolation="step")

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([1.5], "s")
    time_delta_single_q = Q_(1.5, "s")
    value_interp_single = ts_discrete.interp_time(time_delta_single)
    value_interp_single_q = ts_discrete.interp_time(time_delta_single_q)

    assert np.isclose(value_interp_single.data.magnitude, 11)
    assert np.isclose(value_interp_single_q.data.magnitude, 11)

    assert value_interp_single.time == time_delta_single
    assert value_interp_single_q.time == time_delta_single

    assert value_interp_single.data.check(values.dimensionality)
    assert value_interp_single_q.data.check(values.dimensionality)

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
    time_delta_multi_q = Q_([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
    value_interp_multi = ts_discrete.interp_time(time_delta_multi)
    value_interp_multi_q = ts_discrete.interp_time(time_delta_multi_q)

    assert np.all(
        np.isclose(value_interp_multi.data.magnitude, [10, 10, 11, 11, 12, 14, 16, 16])
    )
    assert np.all(
        np.isclose(
            value_interp_multi_q.data.magnitude, [10, 10, 11, 11, 12, 14, 16, 16]
        )
    )

    assert np.all(value_interp_multi.time == time_delta_multi)
    assert np.all(value_interp_multi_q.time == time_delta_multi)

    assert value_interp_multi.data.check(values.dimensionality)
    assert value_interp_multi_q.data.check(values.dimensionality)

    # exceptions ------------------------------------------
    # wrong type
    with pytest.raises(ValueError):
        ts_discrete.interp_time(pd.DatetimeIndex(["2010-10-10"]))
    with pytest.raises(ValueError):
        ts_discrete.interp_time("str")
    with pytest.raises(ValueError):
        ts_discrete.interp_time([1, 2, 3])


def test_time_series_interp_time_discrete_linear():
    """Test the inter_time method for discrete data and linear interpolation."""
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = TimeSeries(data=values, time=time, interpolation="linear")

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([1.5], "s")
    time_delta_single_q = Q_(1.5, "s")
    value_interp_single = ts_discrete.interp_time(time_delta_single)
    value_interp_single_q = ts_discrete.interp_time(time_delta_single_q)

    assert np.isclose(value_interp_single.data.magnitude, 11.5)
    assert np.isclose(value_interp_single_q.data.magnitude, 11.5)

    assert value_interp_single.time == time_delta_single
    assert value_interp_single_q.time == time_delta_single

    assert value_interp_single.data.check(values.dimensionality)
    assert value_interp_single_q.data.check(values.dimensionality)

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([-3, 2.5, 3, 4, 7], "s")
    time_delta_multi_q = Q_([-3, 2.5, 3, 4, 7], "s")
    value_interp_multi = ts_discrete.interp_time(time_delta_multi)
    value_interp_multi_q = ts_discrete.interp_time(time_delta_multi_q)

    assert np.all(np.isclose(value_interp_multi.data.magnitude, [10, 13, 14, 16, 16]))
    assert np.all(np.isclose(value_interp_multi_q.data.magnitude, [10, 13, 14, 16, 16]))

    assert np.all(value_interp_multi.time == time_delta_multi)
    assert np.all(value_interp_multi_q.time == time_delta_multi)

    assert value_interp_multi.data.check(values.dimensionality)
    assert value_interp_multi_q.data.check(values.dimensionality)

    # exceptions ------------------------------------------
    # wrong type
    with pytest.raises(ValueError):
        ts_discrete.interp_time(pd.DatetimeIndex(["2010-10-10"]))
    with pytest.raises(ValueError):
        ts_discrete.interp_time("str")
    with pytest.raises(ValueError):
        ts_discrete.interp_time([1, 2, 3])


def test_time_series_interp_time_expression():
    """Test the TimeSeries.inter_time method for mathematical expressions as data."""
    # needed for TimedeltaIndex tests, since internal conversion introduces small errors
    def _check_close(q1, q2):
        q1 = q1.to_reduced_units()
        q2 = q2.to_reduced_units()

        assert np.all(np.isclose(q1.magnitude, q2.magnitude, atol=1e-9))
        assert q1.units == q2.units

    # scalar ----------------------------------------------
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "meter/second"), "b": Q_(-2, "meter")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = TimeSeries(data=expr)

    # single timedelta
    time_single = Q_(1, "second")
    time_single_pd = pd.TimedeltaIndex([1], unit="s")
    value_interp_single = ts_expr.interp_time(time_single)
    value_interp_single_pd = ts_expr.interp_time(time_single_pd)

    assert value_interp_single.data == Q_(0, "meter")
    _check_close(value_interp_single_pd.data, Q_(0, "meter"))

    # multiple time deltas
    time_multi = Q_([0, 1, 2, 10], "second")
    time_multi_pd = pd.TimedeltaIndex([0, 1, 2, 10], unit="s")
    value_interp_multi = ts_expr.interp_time(time_multi)
    value_interp_multi_pd = ts_expr.interp_time(time_multi_pd)

    assert len(value_interp_multi) == 4
    assert len(value_interp_multi_pd) == 4

    for i in range(4):
        exp = parameters["a"] * time_multi[i] + parameters["b"]
        assert value_interp_multi[i].data == exp
        _check_close(value_interp_multi_pd[i].data, exp)

    # vector -----------------------------------------------
    expr_string_vec = "a*t+b"
    parameters_vec = {"a": Q_([[2, 3, 4]], "1/s"), "b": Q_([[-2, 3, 1]], "")}
    expr_vec = MathematicalExpression(
        expression=expr_string_vec, parameters=parameters_vec
    )

    ts_expr_vec = TimeSeries(data=expr_vec)

    # single time delta
    value_interp_vec_single = ts_expr_vec.interp_time(time_single)
    value_interp_vec_single_pd = ts_expr_vec.interp_time(time_single_pd)

    assert np.all(np.isclose(value_interp_vec_single.data.magnitude, [0, 6, 5]))
    _check_close(value_interp_vec_single_pd.data, Q_([0, 6, 5], ""))

    # multiple time deltas
    value_interp_vec_multi = ts_expr_vec.interp_time(time_multi)
    value_interp_vec_multi_pd = ts_expr_vec.interp_time(time_multi_pd)

    assert value_interp_vec_multi.shape == (4, 3)
    assert value_interp_vec_multi_pd.shape == (4, 3)

    for i in range(4):
        exp = parameters_vec["a"] * time_multi[i] + parameters_vec["b"]
        assert np.all(value_interp_vec_multi[i].data == exp)
        _check_close(value_interp_vec_multi_pd[i].data, exp)

    # exceptions ------------------------------------------
    with pytest.raises(ValueError):
        ts_expr.interp_time(Q_(2, "s/m"))
    with pytest.raises(ValueError):
        ts_expr.interp_time(1)
    with pytest.raises(ValueError):
        ts_expr.interp_time(pd.DatetimeIndex(["2010-10-10"]))
    with pytest.raises(ValueError):
        ts_expr.interp_time("str")
    with pytest.raises(ValueError):
        ts_expr.interp_time([1, 2, 3])
