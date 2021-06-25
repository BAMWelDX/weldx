"""Tests of the core package."""


import numpy as np
import pandas as pd
import pint
import pytest

import weldx.util as ut
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.core import MathematicalExpression, TimeSeries
from weldx.tests._helpers import get_test_name

# --------------------------------------------------------------------------------------
# MathematicalExpression
# --------------------------------------------------------------------------------------


class TestMathematicalExpression:
    """Tests the mathematical expression class."""

    # Fixtures, aliases and shared variables -------------------------------------------

    ME = MathematicalExpression
    # unfortunately, fixtures can not be used in a parametrize section
    expr_def = "(a + b)**2 + c - d"
    params_def = {"a": 2, "c": 3.5}

    @staticmethod
    @pytest.fixture()
    def ma_def() -> MathematicalExpression:
        """Get a default instance for tests."""
        return MathematicalExpression(
            TestMathematicalExpression.expr_def,
            TestMathematicalExpression.params_def,
        )

    # Helper functions -----------------------------------------------------------------

    @staticmethod
    def _check_params_and_vars(expression, exp_params, exp_vars):
        """Check parameters and variables of an MathematicalExpression."""
        assert expression.num_parameters == len(exp_params)
        assert expression.parameters == exp_params

        assert expression.num_variables == len(exp_vars)
        for variable in exp_vars:
            assert variable in expression.get_variable_names()

    # test_construction ----------------------------------------------------------------

    @pytest.mark.parametrize(
        "expression, parameters,  exp_vars",
        [
            ("a*b + c/d - e", {"d": 1, "e": 2}, ["a", "b", "c"]),
            ("a*b + c/d - e", {}, ["a", "b", "c", "d", "e"]),
            ("a**2 + b - c", {"a": 1, "c": 2}, ["b"]),
        ],
    )
    def test_construction(self, expression, parameters, exp_vars):
        """Test the construction."""
        expr = MathematicalExpression(expression=expression, parameters=parameters)

        self._check_params_and_vars(expr, parameters, exp_vars)

    # test_construction_exceptions -----------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "expression, parameters, exception_type, name",
        [
            ("a*b + c/d - e", {"f": 1}, ValueError, "# parameter not in expression"),
            ("a*b + c/d - e", 1, ValueError, "# invalid parameter type"),
            ("a + $b#!==3", {"a": 1}, Exception, "# invalid expression"),
        ],
        ids=get_test_name,
    )
    def test_construction_exceptions(expression, parameters, exception_type, name):
        """Test the exceptions of the '__init__' method."""
        with pytest.raises(exception_type):
            MathematicalExpression(expression=expression, parameters=parameters)

    # test_set_parameter ---------------------------------------------------------------

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

    # test_set_parameter_exceptions ----------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "name, value, exception_type, test_name",
        [
            ("k", 1, ValueError, "# parameter not in expression"),
            (33, 1, TypeError, "# wrong type as name #1"),
            ({"a": 1}, 1, TypeError, "# wrong type as name #2"),
        ],
        ids=get_test_name,
    )
    def test_set_parameter_exceptions(ma_def, name, value, exception_type, test_name):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ma_def.set_parameter(name, value)

    # test_comparison ------------------------------------------------------------------

    expr_mat_identical = "a**2 + 2*a*b + b**2 + c - d"
    expr_different = "a*b + c*d"
    params_too_many = {"a": 2, "c": 3.5, "d": 4}
    params_wrong_value = {"a": 2, "c": 1.5}

    @staticmethod
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
        ma_def,
        other,
        equal,
        equal_no_params,
        mat_equal,
        mat_equal_no_params,
    ):
        """Test if another object is equal to the default instance."""
        assert (ma_def == other) is equal
        assert (ma_def != other) is not equal
        assert ma_def.equals(other, False, True) is equal_no_params
        assert ma_def.equals(other) is mat_equal
        assert ma_def.equals(other, False, False) is mat_equal_no_params

    # -----------------------------------------------------

    # TODO: Add tests for quantities
    @staticmethod
    @pytest.mark.parametrize(
        "expression, parameters, variables, exp_result",
        [
            ("a*b + c/d - e", {"d": 1, "e": 2}, {"a": 1, "b": 2, "c": 3}, 3),
            ("(a + b)**2 + c - d", {"a": 3, "d": 2}, {"b": 2, "c": 4}, 27),
            (
                "a + b",
                {"a": np.array([1, 2])},
                {"b": np.array([2, 4])},
                [3, 6],
            ),
        ],
    )
    def test_evaluation(expression, parameters, variables, exp_result):
        """Test the evaluation of the mathematical function."""
        expr = MathematicalExpression(expression=expression, parameters=parameters)

        assert np.all(expr.evaluate(**variables) == exp_result)

    # test_evaluate_exceptions ---------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "variables, exception_type, test_name",
        [
            ({"b": 1, "c": 2, "d": 3}, ValueError, "# input is expression parameter"),
            ({"b": 1}, Exception, "# not enough values provided"),
        ],
        ids=get_test_name,
    )
    def test_evaluate_exceptions(ma_def, variables, exception_type, test_name):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ma_def.evaluate(**variables)


# --------------------------------------------------------------------------------------
# TimeSeries
# --------------------------------------------------------------------------------------


class TestTimeSeries:
    """Tests for the TimeSeries class."""

    # Fixtures, aliases and shared variables -------------------------------------------

    ME = MathematicalExpression
    DTI = pd.DatetimeIndex
    TDI = pd.TimedeltaIndex
    TS = TimeSeries

    time_discrete = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    value_constant = Q_(1, "m")
    values_discrete = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    me_expr_str = "a*t + b"
    me_params = {"a": Q_(2, "m/s"), "b": Q_(-2, "m")}

    me_params_vec = {"a": Q_([[2, 0, 1]], "m/s"), "b": Q_([[-2, 3, 0]], "m")}

    ts_constant = TimeSeries(value_constant)
    ts_disc_step = TimeSeries(values_discrete, time_discrete, "step")
    ts_disc_linear = TimeSeries(values_discrete, time_discrete, "linear")
    ts_expr = TimeSeries(ME(me_expr_str, me_params))
    ts_expr_vec = TimeSeries(ME(me_expr_str, me_params_vec))

    # test_construction_discrete -------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "data, time, interpolation, shape_exp",
        [
            (Q_(1, "m"), None, None, (1,)),
            (Q_([3, 7, 1], "m"), TDI([0, 1, 2], unit="s"), "step", (3,)),
            (Q_([3, 7, 1], ""), Q_([0, 1, 2], "s"), "step", (3,)),
        ],
    )
    def test_construction_discrete(data, time, interpolation, shape_exp):
        """Test the construction of the TimeSeries class."""
        # set expected values
        time_exp = time
        if isinstance(time_exp, pint.Quantity):
            time_exp = pd.TimedeltaIndex(time_exp.m, unit="s")

        exp_interpolation = interpolation
        if len(data.m.shape) == 0 and interpolation is None:
            exp_interpolation = "step"

        # create instance
        ts = TimeSeries(data=data, time=time, interpolation=interpolation)

        # check
        assert np.all(ts.data == data)
        assert np.all(ts.time == time_exp)
        assert ts.interpolation == exp_interpolation
        assert ts.shape == shape_exp
        assert data.check(UREG.get_dimensionality(ts.units))

        assert np.all(ts.data_array.data == data)
        assert ts.data_array.attrs["interpolation"] == exp_interpolation
        if time_exp is None:
            assert "time" not in ts.data_array
        else:
            assert np.all(ts.data_array.time == time_exp)

    # test_construction_expression -----------------------------------------------------

    params_scalar = {"a": Q_(2, "1/s"), "b": Q_(-2, "")}
    params_vec = {"a": Q_([[2, 3, 4]], "m/s"), "b": Q_([[-2, 3, 1]], "m")}

    @staticmethod
    @pytest.mark.parametrize(
        "data,  shape_exp, unit_exp",
        [
            (ME("a*t + b", params_scalar), (1,), ""),
            (ME("a*t + b", params_vec), (1, 3), "m"),
        ],
    )
    def test_construction_expression(data, shape_exp, unit_exp):
        """Test the construction of the TimeSeries class."""
        ts = TimeSeries(data=data)

        # check
        assert ts.data == data
        assert ts.time is None
        assert ts.interpolation is None
        assert ts.shape == shape_exp
        assert ts.data_array is None
        assert Q_(1, unit_exp).check(UREG.get_dimensionality(ts.units))

    # test_construction_exceptions -----------------------------------------------------

    values_def = Q_([5, 7, 3, 6, 8], "m")
    time_def = Q_([0, 1, 2, 3, 4], "s")
    me_too_many_vars = ME("a*t + b", {})
    me_param_units = ME("a*t + b", {"a": Q_(2, "1/s"), "b": Q_(-2, "m")})
    me_time_vec = ME("a*t + b", {"a": Q_([2, 3, 4], "1/s"), "b": Q_([-2, 3, 1], "")})

    @staticmethod
    @pytest.mark.parametrize(
        "data, time, interpolation, exception_type, test_name",
        [
            (values_def, time_def, "int", ValueError, "# unknown interpolation"),
            (values_def, time_def.magnitude, "step", ValueError, "# invalid time type"),
            (me_too_many_vars, None, None, Exception, "# too many free variables"),
            (me_param_units, None, None, Exception, "# incompatible parameter units"),
            (me_time_vec, None, None, Exception, "# not compatible with time vectors"),
            ("a string", None, None, TypeError, "# wrong data type"),
        ],
        ids=get_test_name,
    )
    def test_construction_exceptions(
        data, time, interpolation, exception_type, test_name
    ):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            TimeSeries(data=data, time=time, interpolation=interpolation)

    # test_comparison -------------------------------------

    time_wrong_values = TDI([0, 1, 2, 3, 5], unit="s")
    values_discrete_wrong = Q_(np.array([10, 11, 12, 15, 16]), "mm")
    values_unit_wrong = Q_(np.array([10, 11, 12, 14, 16]), "s")
    values_unit_prefix_wrong = Q_(np.array([10, 11, 12, 14, 16]), "m")
    params_wrong_values = {"a": Q_(2, "1/s"), "b": Q_(-1, "")}
    params_wrong_unit = {"a": Q_(2, "g/s"), "b": Q_(-2, "g")}
    params_wrong_unit_prefix = {"a": Q_(2, "m/ms"), "b": Q_(-2, "m")}

    @staticmethod
    @pytest.mark.parametrize(
        "ts, ts_other, result_exp",
        [
            (ts_constant, TS(value_constant), True),
            (ts_disc_step, TS(values_discrete, time_discrete, "step"), True),
            (ts_expr, TS(ME(me_expr_str, me_params)), True),
            (ts_constant, ts_disc_step, False),
            (ts_constant, ts_expr, False),
            (ts_disc_step, ts_expr, False),
            (ts_constant, 1, False),
            (ts_disc_step, 1, False),
            (ts_expr, 1, False),
            (ts_constant, "wrong", False),
            (ts_disc_step, "wrong", False),
            (ts_expr, "wrong", False),
            (ts_constant, TS(Q_(1337, "m")), False),
            (ts_constant, TS(Q_(1, "mm")), False),
            (ts_constant, TS(Q_(1, "s")), False),
            (ts_disc_step, TS(values_discrete, time_wrong_values, "step"), False),
            (ts_disc_step, TS(values_discrete_wrong, time_discrete, "step"), False),
            (ts_disc_step, TS(values_unit_prefix_wrong, time_discrete, "step"), False),
            (ts_disc_step, TS(values_discrete, time_discrete, "linear"), False),
            (ts_expr, TS(ME("a*t + 2*b", me_params)), False),
            (ts_expr, TS(ME(me_expr_str, params_wrong_values)), False),
            (ts_expr, TS(ME(me_expr_str, params_wrong_unit)), False),
            (ts_expr, TS(ME(me_expr_str, params_wrong_unit_prefix)), False),
        ],
    )
    def test_comparison(ts, ts_other, result_exp):
        """Test the TimeSeries comparison methods."""
        assert (ts == ts_other) is result_exp
        assert (ts != ts_other) is not result_exp

    # test_interp_time -----------------------------------------------------------------

    time_single = pd.TimedeltaIndex([2.1], "s")
    time_single_q = Q_(2.1, "s")
    time_mul = pd.TimedeltaIndex([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
    time_mul_q = Q_([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
    results_exp_vec = [
        [-8, 3, -3],
        [-0.6, 3, 0.7],
        [0.2, 3, 1.1],
        [1.8, 3, 1.9],
        [3, 3, 2.5],
        [4, 3, 3],
        [6, 3, 4],
        [12, 3, 7],
    ]

    @staticmethod
    @pytest.mark.parametrize(
        "ts, time, magnitude_exp, unit_exp",
        [
            (ts_constant, time_single, 1, "m"),
            (ts_constant, time_single_q, 1, "m"),
            (ts_constant, time_mul, [1, 1, 1, 1, 1, 1, 1, 1], "m"),
            (ts_constant, time_mul_q, [1, 1, 1, 1, 1, 1, 1, 1], "m"),
            (ts_disc_step, time_single, 12, "mm"),
            (ts_disc_step, time_single_q, 12, "mm"),
            (ts_disc_step, time_mul, [10, 10, 11, 11, 12, 14, 16, 16], "mm"),
            (ts_disc_step, time_mul_q, [10, 10, 11, 11, 12, 14, 16, 16], "mm"),
            (ts_disc_linear, time_single, 12.2, "mm"),
            (ts_disc_linear, time_single_q, 12.2, "mm"),
            (ts_disc_linear, time_mul, [10, 10.7, 11.1, 11.9, 13, 14, 16, 16], "mm"),
            (ts_disc_linear, time_mul_q, [10, 10.7, 11.1, 11.9, 13, 14, 16, 16], "mm"),
            (ts_expr, time_single, 2.2, "m"),
            (ts_expr, time_single_q, 2.2, "m"),
            (ts_expr, time_mul, [-8, -0.6, 0.2, 1.8, 3, 4, 6, 12], "m"),
            (ts_expr, time_mul_q, [-8, -0.6, 0.2, 1.8, 3, 4, 6, 12], "m"),
            (ts_expr_vec, time_single, [[2.2, 3, 2.1]], "m"),
            (ts_expr_vec, time_single_q, [[2.2, 3, 2.1]], "m"),
            (ts_expr_vec, time_mul, results_exp_vec, "m"),
        ],
    )
    def test_interp_time(ts, time, magnitude_exp, unit_exp):
        """Test the interp_time function."""
        result = ts.interp_time(time)

        assert np.all(np.isclose(result.data.magnitude, magnitude_exp))
        assert Q_(1, str(result.units)) == Q_(1, unit_exp)

        exp_time = time
        if isinstance(exp_time, pint.Quantity):
            exp_time = ut.to_pandas_time_index(time)
        if len(exp_time) == 1:
            exp_time = None

        assert np.all(result.time == exp_time)

    # test_interp_time_warning ---------------------------------------------------------

    @staticmethod
    def test_interp_time_warning():
        """Test if a warning is emitted when interpolating already interpolated data."""
        ts = TimeSeries(data=Q_([1, 2, 3], "m"), time=Q_([0, 1, 2], "s"))
        with pytest.warns(None) as recorded_warnings:
            ts_interp = ts.interp_time(Q_([0.25, 0.5, 0.75, 1], "s"))
        assert len(recorded_warnings) == 0

        with pytest.warns(UserWarning):
            ts_interp.interp_time(Q_([0.4, 0.6], "s"))

    # test_interp_time_exceptions ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("ts", [ts_constant, ts_disc_step, ts_disc_linear, ts_expr])
    @pytest.mark.parametrize(
        "time,  exception_type, test_name",
        [
            (DTI(["2010-10-10"]), ValueError, "# wrong type #1"),
            ("a string", ValueError, "# wrong type #2"),
            ([1, 2, 3], ValueError, "# wrong type #3"),
            (1, ValueError, "# wrong type #4"),
            (Q_(2, "s/m"), Exception, "# wrong type #5"),
        ],
        ids=get_test_name,
    )
    def test_interp_time_exceptions(ts, time, exception_type, test_name):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ts.interp_time(time)
