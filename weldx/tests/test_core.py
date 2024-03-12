"""Tests of the core package."""

import warnings

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
from xarray import DataArray

from weldx.constants import Q_, U_
from weldx.core import GenericSeries, MathematicalExpression, TimeSeries
from weldx.tests._helpers import get_test_name
from weldx.time import Time

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
            (33, 1, ValueError, "# wrong type as name #1"),
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

    @staticmethod
    @pytest.mark.slow
    def test_integrate_length_computation():
        """Ensure we can integrate with Sympy during length computation."""
        from weldx import DynamicShapeSegment

        class MySegment(DynamicShapeSegment):
            def __init__(self):
                f = "x * sin(s) + y * s"
                p = dict(x=Q_([1, 0, 0], "mm"), y=Q_([0, 1, 0], "mm"))
                super().__init__(f, parameters=p)

        s = MySegment()
        assert s.get_section_length(1).u == Q_("mm")


# --------------------------------------------------------------------------------------
# TimeSeries
# --------------------------------------------------------------------------------------


class TestTimeSeries:
    """Tests for the TimeSeries class."""

    # Fixtures, aliases and shared variables -------------------------------------------

    ME = MathematicalExpression
    DTI = pd.DatetimeIndex
    TS = TimeSeries

    time_discrete = pd.to_timedelta([0, 1, 2, 3, 4], "s")
    value_constant = Q_(1, "m")
    values_discrete = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    me_expr_str = "a*t + b"
    me_params = {"a": Q_(2, "m/s"), "b": Q_(-2, "m")}

    me_params_vec = {"a": Q_([2, 0, 1], "m/s"), "b": Q_([-2, 3, 0], "m")}

    ts_const = TimeSeries(value_constant)
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
            (Q_([3, 7, 1], "m"), pd.to_timedelta([0, 1, 2], unit="s"), "step", (3,)),
            (Q_([3, 7, 1], ""), Q_([0, 1, 2], "s"), "step", (3,)),
            (Q_([3, 7, 1], ""), DTI(["2010", "2011", "2012"]), "step", (3,)),
        ],
    )
    @pytest.mark.parametrize("reference_time", [None, "2000-01-01"])
    def test_construction_discrete(
        data: pint.Quantity, time, interpolation, shape_exp, reference_time
    ):
        """Test the construction of the TimeSeries class."""
        if reference_time is not None and isinstance(time, (pd.DatetimeIndex)):
            pytest.skip()

        # set expected values
        time_exp = time

        if time_exp is not None:
            time_exp = Time(time, reference_time)

        exp_interpolation = interpolation
        if len(data.shape) == 0 and interpolation is None:
            exp_interpolation = "step"

        # create instance
        ts = TimeSeries(
            data=data,
            time=time,
            interpolation=interpolation,
            reference_time=reference_time,
        )

        # check
        assert np.all(ts.data == data)
        if time_exp is not None:
            assert ts.reference_time == time_exp.reference_time
            assert ts.time.all_close(time_exp)
        assert ts.interpolation == exp_interpolation
        assert ts.shape == shape_exp
        assert data.is_compatible_with(ts.units)

        assert np.all(ts.data_array.data == data)
        assert ts.data_array.attrs["interpolation"] == exp_interpolation
        if time_exp is None:
            assert "time" not in ts.data_array
        else:
            assert Time(ts.data_array.time).all_close(time_exp)

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
        assert U_(unit_exp).is_compatible_with(ts.units)

    # test_init_data_array -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "data, dims, coords, exception_type",
        [
            (Q_([1, 2, 3], "m"), "time", dict(time=pd.to_timedelta([1, 2, 3])), None),
            (Q_([1, 2, 3], "m"), "a", dict(a=pd.to_timedelta([1, 2, 3])), KeyError),
            (
                Q_([[1, 2]], "m"),
                ("a", "time"),
                dict(a=[2], time=pd.to_timedelta([1, 2])),
                None,
            ),
            (Q_([1, 2, 3], "m"), "time", None, KeyError),
            (Q_([1, 2, 3], "m"), "time", dict(time=[1, 2, 3]), TypeError),
            ([1, 2, 3], "time", dict(time=pd.to_timedelta([1, 2, 3])), TypeError),
        ],
    )
    @pytest.mark.parametrize("reference_time", [None, "2000-01-01"])
    def test_init_data_array(data, dims, coords, reference_time, exception_type):
        """Test the `__init__` method with an xarray as data parameter."""
        da = xr.DataArray(data=data, dims=dims, coords=coords)
        exp_time_ref = None
        if reference_time is not None:
            da.weldx.time_ref = reference_time
            exp_time_ref = pd.Timestamp(reference_time)

        if exception_type is not None:
            with pytest.raises(exception_type):
                TimeSeries(da)
        else:
            ts = TimeSeries(da)
            assert ts.data_array.dims[0] == "time"
            assert ts.reference_time == exp_time_ref

    # test_construction_exceptions -----------------------------------------------------

    values_def = Q_([5, 7, 3, 6, 8], "m")
    time_def = Q_([0, 1, 2, 3, 4], "s")
    me_too_many_vars = ME("a*t + b", {})
    me_param_units = ME("a*t + b", {"a": Q_(2, "1/s"), "b": Q_(-2, "m")})

    @staticmethod
    @pytest.mark.parametrize(
        "data, time, interpolation, exception_type, test_name",
        [
            (values_def, time_def, "int", ValueError, "# unknown interpolation"),
            (values_def, time_def.magnitude, "step", TypeError, "# invalid time type"),
            (me_too_many_vars, None, None, Exception, "# too many free variables"),
            (me_param_units, None, None, Exception, "# incompatible parameter units"),
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

    time_wrong_values = pd.to_timedelta([0, 1, 2, 3, 5], "s")
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
            (ts_const, TS(value_constant), True),
            (ts_disc_step, TS(values_discrete, time_discrete, "step"), True),
            (ts_expr, TS(ME(me_expr_str, me_params)), True),
            (ts_const, ts_disc_step, False),
            (ts_const, ts_expr, False),
            (ts_disc_step, ts_expr, False),
            (ts_const, 1, False),
            (ts_disc_step, 1, False),
            (ts_expr, 1, False),
            (ts_const, "wrong", False),
            (ts_disc_step, "wrong", False),
            (ts_expr, "wrong", False),
            (ts_const, TS(Q_(1337, "m")), False),
            (ts_const, TS(Q_(1, "mm")), False),
            (ts_const, TS(Q_(1, "s")), False),
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

    time_single = pd.to_timedelta([2.1], "s")
    time_single_q = Q_(2.1, "s")
    time_mul = pd.to_timedelta([-3, 0.7, 1.1, 1.9, 2.5, 3, 4, 7], "s")
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
            (ts_const, time_single, 1, "m"),
            (ts_const, time_single_q, 1, "m"),
            (ts_const, time_mul, [1, 1, 1, 1, 1, 1, 1, 1], "m"),
            (ts_const, time_mul + pd.Timestamp("2020"), [1, 1, 1, 1, 1, 1, 1, 1], "m"),
            (ts_const, time_mul_q, [1, 1, 1, 1, 1, 1, 1, 1], "m"),
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
    @pytest.mark.parametrize("reference_time", [None, "2000-01-01"])
    def test_interp_time(ts, time, magnitude_exp, unit_exp, reference_time):
        """Test the interp_time function."""
        if reference_time is not None:
            if isinstance(ts.data, xr.DataArray):
                ts = TimeSeries(
                    ts.data_array,
                    reference_time=reference_time,
                    interpolation=ts.interpolation,
                )
            else:
                ts = TimeSeries(
                    ts.data,
                    time=ts.time,
                    reference_time=reference_time,
                    interpolation=ts.interpolation,
                )
            time = Time(time, time_ref=reference_time)

        result = ts.interp_time(time)

        assert np.all(np.isclose(result.data.magnitude, magnitude_exp))
        assert result.units == U_(unit_exp)

        time = Time(time, reference_time)
        if len(time) == 1:
            assert result.time is None
        else:
            assert result.time.all_close(time)

    # test_interp_time_warning ---------------------------------------------------------

    @staticmethod
    def test_interp_time_warning():
        """Test if a warning is emitted when interpolating already interpolated data."""
        ts = TimeSeries(data=Q_([1, 2, 3], "m"), time=Q_([0, 1, 2], "s"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            ts_interp = ts.interp_time(Q_([0.25, 0.5, 0.75, 1], "s"))

        with pytest.warns(UserWarning):
            ts_interp.interp_time(Q_([0.4, 0.6], "s"))

    # test_interp_time_exceptions ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("ts", [ts_const, ts_disc_step, ts_disc_linear, ts_expr])
    @pytest.mark.parametrize(
        "time,  exception_type, test_name",
        [
            # (DTI(["2010-10-10"]), ValueError, "# wrong type #1"),  # skipcq: PY-W0069
            ("a string", TypeError, "# wrong type #2"),
            ([1, 2, 3], TypeError, "# wrong type #3"),
            (1, TypeError, "# wrong type #4"),
            (Q_(2, "s/m"), Exception, "# wrong type #5"),
        ],
        ids=get_test_name,
    )
    def test_interp_time_exceptions(ts, time, exception_type, test_name):
        """Test the exceptions of the 'set_parameter' method."""
        with pytest.raises(exception_type):
            ts.interp_time(time)


# --------------------------------------------------------------------------------------
# GenericSeries
# --------------------------------------------------------------------------------------


# todo
#   - check update of variable units to unitless
#   - test partial evaluation


class TestGenericSeries:
    """Test the `GenericSeries`."""

    # test_init_discrete ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "data_units, dims, coordinates",
        [
            ("V", ["u", "v"], dict(u=Q_([0, 1, 2], "m"), v=Q_([7, 8], "A"))),
        ],
    )
    def test_init_discrete(data_units, dims, coordinates):
        data_shape = tuple(len(v) for v in coordinates.values())
        data = Q_(np.ones(data_shape), data_units)
        gs = GenericSeries(data, dims, coordinates)

        assert GenericSeries(gs.data_array) == gs

    # test_init_expression -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "dims, units, parameters, exception",
        [
            # 4 dims without units - 0 params
            (None, None, None, None),
            # 4 dims without units - 0 params - custom dimension names
            (dict(a="d1", x="d2", b="d3", y="d4"), None, None, None),
            # ERROR - dims with identical dimension name
            (dict(a="d1", x="d2", b="d1", y="d4"), None, None, ValueError),
            # 4 dims with units - 0 params
            (None, dict(a="m", x="m", b="K", y="m*m/K"), None, None),
            # ERROR - 4 dims with incompatible units - 0 params
            (None, dict(a="m", x="m", b="K", y="m/K"), None, pint.DimensionalityError),
            # 2 dims with units - 2 scalar params
            (None, dict(x="m", y="m*m/K"), dict(a="3m", b="30K"), None),
            # ERROR - parameter and variable units are incompatible
            (None, dict(b="m", y="m"), dict(a="3m", x="30K"), pint.DimensionalityError),
            # 3 dims with units - 1 array parameter (quantity)
            (None, None, dict(y=Q_([1, 2], "m")), None),
            # 3 dims with units - 1 array parameter (tuple)
            (None, None, dict(y=(Q_([1, 2], "m"), "d1")), None),
            # 3 dims with units - 1 array parameter (DataArray)
            (None, None, dict(y=DataArray(Q_([1, 2], "m"))), None),
            # ERROR - expression has no variables
            (None, None, dict(a="3m", x="4m", b="5m", y="6m"), ValueError),
            # ERROR - Parameter dimension is also a variable dimension (tuple)
            (None, None, dict(y=(Q_([1, 2], "m"), "a")), ValueError),
            # ERROR - Parameter dimension is also a variable dimension (DataArray)
            (None, None, dict(y=DataArray(Q_([1, 2], "m"), dims=["x"])), ValueError),
            # ERROR - Same parameter dimensions of different sizes
            (
                None,
                None,
                dict(x=(Q_([1, 2], "m"), "a"), y=(Q_([1, 2, 3], "m"), "a")),
                ValueError,
            ),
        ],
    )
    def test_init_expression(dims, units, parameters, exception):
        if units is None and parameters is not None:
            units = {k: "m" for k in ["a", "b", "x", "y"] if k not in parameters.keys()}

        expr = "a*x + b*y"
        if exception is not None:
            with pytest.raises(exception):
                GenericSeries(expr, dims=dims, parameters=parameters, units=units)
            return

        gs = GenericSeries(expr, dims=dims, parameters=parameters, units=units)

        assert GenericSeries(gs.data, dims=dims, units=units) == gs

    # test_call_operator_discrete ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "u,v,w",
        [
            ("0.25m", "1.5K", "0A"),
            (Q_([0, 2], "m"), Q_([1, 2, 5], "K"), Q_([1, 2], "A")),
        ],
    )
    def test_call_operator_discrete(u, v, w):
        # setup
        data = Q_(np.array(range(2 * 3 * 4)).reshape((2, 3, 4)), "V")
        dims = ["u", "v", "w"]
        coords = dict(u=Q_([0, 1], "m"), v=Q_([0, 1, 2], "K"), w=Q_([0, 1, 2, 3], "A"))

        params = dict(u=u, v=v, w=w)
        gs = GenericSeries(data, dims, coords)

        # perform interpolation
        gs_interp = gs(**params)

        # calculate expected results
        params = {k: Q_(v) for k, v in params.items()}
        for k, val in params.items():
            if len(val.shape) == 0:
                params[k] = np.expand_dims(val, 0)
        exp_shape = tuple(len(v.m) for v in params.values())
        exp_data = np.zeros(exp_shape)

        for i, u_v in enumerate(params["u"]):
            for j, v_v in enumerate(params["v"]):
                for k, w_v in enumerate(params["w"]):
                    exp_data[i, j, k] = (
                        np.clip(u_v.m, 0, 1) * 3 * 4
                        + np.clip(v_v.m, 0, 2) * 4
                        + np.clip(w_v.m, 0, 3)
                    )

        # check results
        assert np.allclose(gs_interp.data, Q_(exp_data, "V"))

    # test_call_operator_expression ----------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "u,v,w",
        [
            ("1m", "8K", "10m**2"),
            (Q_([1, 2, 3], "m"), Q_([1, 2, 3, 5], "K"), Q_([1, 2], "m*m")),
        ],
    )
    def test_call_operator_expression(u, v, w):
        # setup
        expression = "a*u + b*v + w"
        units = dict(u="m", v="K", w="m*m")
        parameters = dict(a="2m", b="5m*m/K")

        params = dict(u=u, v=v, w=w)
        gs = GenericSeries(expression, parameters=parameters, units=units)

        # perform interpolation
        gs_interp = gs(**params)

        # calculate expected result
        params = {k: Q_(val) for k, val in params.items()}
        for k, val in params.items():
            if len(val.shape) == 0:
                params[k] = np.expand_dims(val, 0)
        exp_shape = tuple(len(val) for val in params.values())
        exp_data = np.zeros(exp_shape)

        a = Q_(parameters["a"])
        b = Q_(parameters["b"])

        for i, u_v in enumerate(params["u"]):
            for j, v_v in enumerate(params["v"]):
                for k, w_v in enumerate(params["w"]):
                    exp_data[i, j, k] = (a * u_v + b * v_v + w_v).m

        assert np.allclose(gs_interp.data, Q_(exp_data, "m*m"))

    # todo:
    #  - 2d variables not allowed
    #  - test evaluation of expression with renamed dims/variables


# --------------------------------------------------------------------------------------
# Test series that are derived from a GenericSeries
# --------------------------------------------------------------------------------------


class TestDerivedFromGenericSeries:
    @staticmethod
    @pytest.mark.parametrize(
        "expr, exception",
        [
            ("a*b", None),
            ("a*2", None),
            ("2*b", None),
            ("a*y", ValueError),
            ("x*b", ValueError),
            ("x*b", ValueError),
            ("x*y", ValueError),
            ("a*b*c", ValueError),
        ],
    )
    def test_allowed_variables(expr, exception):
        """Test the allowed variables constraints."""

        class _DerivedSeries(GenericSeries):
            _allowed_variables = ["a", "b"]

        if exception is None:
            _DerivedSeries(expr)
            return

        with pytest.raises(exception):
            _DerivedSeries(expr)

    @staticmethod
    @pytest.mark.parametrize(
        "expr, exception",
        [
            ("a*b", None),
            ("a*2", ValueError),
            ("2*b", ValueError),
            ("a*y", ValueError),
            ("x*b", ValueError),
            ("x*b", ValueError),
            ("x*y", ValueError),
            ("a*b*c", None),
        ],
    )
    def test_required_variables(expr, exception):
        """Test the required variables constraints."""

        class _DerivedSeries(GenericSeries):
            _required_variables = ["a", "b"]

        if exception is None:
            _DerivedSeries(expr)
            return

        with pytest.raises(exception):
            _DerivedSeries(expr)

    @staticmethod
    @pytest.mark.parametrize("use_expr", [True, False])
    def test_evaluation_preprocessor_expression(use_expr):
        """Test the evaluation preprocessor."""

        class _DerivedSeries(GenericSeries):
            _evaluation_preprocessor = dict(a=lambda x: 2 * x, b=lambda x: 5 * x)

        if use_expr:
            ds = _DerivedSeries("a+b")
        else:
            ds = _DerivedSeries(
                Q_([[2, 21], [11, 30]]),
                dims=["a", "b"],
                coords=dict(a=Q_([1, 10]), b=Q_([1, 20])),
            )

        result = ds(a=Q_([1, 3, 4]), b=Q_([3, 2, 1]))

        exp_result = [[17, 12, 7], [21, 16, 11], [23, 18, 13]]
        assert np.allclose(exp_result, result.data.m)

    @staticmethod
    @pytest.mark.parametrize(
        "expr, dims, exception",
        [
            ("a*b", None, None),
            ("a*b", dict(a="x"), ValueError),
            ("a*b", dict(b="y"), ValueError),
            ("a*b", dict(a="x", b="y"), ValueError),
            ("a*2", None, ValueError),
            ("2*b", None, ValueError),
            ("a*y", None, ValueError),
            ("x*b", None, ValueError),
            ("x*b", dict(x="a"), None),
            ("x*b", dict(x="b"), ValueError),
            ("x*y", None, ValueError),
            ("x*y", dict(x="a"), ValueError),
            ("x*y", dict(y="b"), ValueError),
            ("x*y", dict(x="a", y="b"), None),
            ("x*y", dict(x="b", y="a"), None),
            ("a*b*c", None, None),
        ],
    )
    def test_required_dimensions_expression(expr, dims, exception):
        """Test required dimension constraint for expression based `GenericSeries`"""

        class _DerivedSeries(GenericSeries):
            _required_dimensions = ["a", "b"]

        if exception is None:
            _DerivedSeries(expr, dims)
            return

        with pytest.raises(exception):
            _DerivedSeries(expr, dims)

    @staticmethod
    @pytest.mark.parametrize(
        "data, dims, exception",
        [
            (Q_(np.zeros((2, 3))), ["a", "b"], None),
            (Q_(np.zeros((2, 3, 4))), ["a", "b", "c"], None),
            (Q_(np.zeros((2, 3))), ["a", "y"], ValueError),
            (Q_(np.zeros((2, 3))), ["x", "b"], ValueError),
            (Q_(np.zeros((2, 3))), ["x", "y"], ValueError),
        ],
    )
    @pytest.mark.parametrize("pass_data_array", [False, True])
    def test_required_dimensions_discrete(data, dims, exception, pass_data_array):
        """Test required dimension constraint for discrete `GenericSeries`"""

        class _DerivedSeries(GenericSeries):
            _required_dimensions = ["a", "b"]

        if pass_data_array:
            data = DataArray(data, dims=dims)
            dims = None

        if exception is None:
            _DerivedSeries(data, dims)
            return

        with pytest.raises(exception):
            _DerivedSeries(data, dims)

    @staticmethod
    @pytest.mark.parametrize(
        "expr, units, parameters, exception",
        [
            ("t*b", None, None, None),
            ("t*b", dict(t="s"), None, None),
            ("t*b", dict(b="m"), None, None),
            ("t*b", dict(t="s", b="m"), None, None),
            ("t*b", dict(t="m"), None, pint.DimensionalityError),
            ("t*b", dict(t="m", b="m"), None, pint.DimensionalityError),
            ("t*b", None, dict(t="3s"), None),
            ("a*t + b", None, None, pint.DimensionalityError),
            ("a*t + b", dict(a="m/s"), None, None),
            ("a*t + b", dict(a="m"), None, pint.DimensionalityError),
            ("a*t + b", None, None, pint.DimensionalityError),
        ],
    )
    def test_required_dimension_units_expression(expr, units, parameters, exception):
        """Test required dimension units constraint for expr.-based `GenericSeries`"""

        class _DerivedSeries(GenericSeries):
            _required_dimension_units = dict(t="s", b="m")

        if exception is None:
            _DerivedSeries(expr, units=units, parameters=parameters)
            return

        with pytest.raises(exception):
            _DerivedSeries(expr, units=units, parameters=parameters)

    @staticmethod
    @pytest.mark.parametrize(
        "dims, units, exception",
        [
            (["t", "b"], dict(t="s", b="m"), None),
            (["t", "b"], dict(t="m"), (pint.DimensionalityError, KeyError)),
            (["t", "b"], dict(t="m", b="m"), pint.DimensionalityError),
            (["t", "b"], dict(b="m"), KeyError),
            (["t", "b"], dict(t="s", b="s"), pint.DimensionalityError),
            (["t", "b"], dict(t="m", b="s"), pint.DimensionalityError),
            (["t"], dict(t="s"), None),
            (["t"], dict(t="m"), pint.DimensionalityError),
            (["b"], dict(b="m"), None),
            (["b"], dict(b="s"), pint.DimensionalityError),
            (["t", "b", "d"], dict(t="s", b="m"), None),
            (["t", "b", "d"], dict(t="s", b="m", d="A"), None),
            (["t", "b", "d"], dict(t="s", b="m", d="m"), None),
            (["t", "d"], dict(t="s", d="m"), None),
            (["t", "d"], dict(t="A", d="m"), pint.DimensionalityError),
            (["b", "d"], dict(b="m", d="m"), None),
            (["b", "d"], dict(b="s", d="m"), pint.DimensionalityError),
        ],
    )
    def test_required_dimension_units_discrete(dims, units, exception):
        """Test required dimension units constraint for discrete `GenericSeries`"""

        class _DerivedSeries(GenericSeries):
            _required_dimension_units = dict(t="s", b="m")

        def _dim_length(i, d, c):
            if d in c:
                return len(c[d])
            return i + 10

        coords = {}
        for i, (k, v) in enumerate(units.items()):
            coords[k] = Q_(np.zeros(i + 2), v)

        shape = tuple(_dim_length(i, d, coords) for (i, d) in enumerate(dims))
        data = Q_(np.ones(shape))

        if exception is None:
            _DerivedSeries(data, dims, coords)
            return

        with pytest.raises(exception):
            _DerivedSeries(data, dims, coords)

    @staticmethod
    @pytest.mark.parametrize(
        "coords, exception",
        [
            (dict(a=["x", "y"]), None),
            (dict(a=["x", "y"], b=[4, 5, 6, 7]), None),
            (dict(b=[4, 5, 6, 7]), KeyError),
            (dict(b=["x", "y", "a", "b"]), KeyError),
            (dict(a=["x", "b"]), ValueError),
            (dict(a=["a", "b"]), ValueError),
            (dict(a=["a", "y"]), ValueError),
            (dict(a=["a", "y"]), ValueError),
            (dict(a=["x", "y", "z"]), ValueError),  # should fail or pass? -> crt fail
        ],
    )
    def test_required_dimension_coordinates_discrete(coords, exception):
        class _DerivedSeries(GenericSeries):
            _required_dimension_coordinates = dict(a=["x", "y"])

        num_coords = 2
        if "a" in coords:
            num_coords = len(coords["a"])

        data = Q_(np.ones((num_coords, 4)))
        dims = ["a", "b"]
        if exception is None:
            _DerivedSeries(data, dims, coords)
            return

        with pytest.raises(exception):
            _DerivedSeries(data, dims, coords)

    @staticmethod
    @pytest.mark.parametrize(
        "coords, exception",
        [
            (dict(a=["x", "y"]), None),
            (dict(a=["x", "y"], b=[4, 5, 6, 7]), None),
            (dict(b=[4, 5, 6, 7]), ValueError),
            (dict(a=["x", "b"]), ValueError),
            (dict(a=["a", "b"]), ValueError),
            (dict(a=["a", "y"]), ValueError),
            (dict(a=["a", "y"]), ValueError),
            (dict(a=["x", "y", "z"]), ValueError),  # should fail or pass? -> crt fail
        ],
    )
    def test_required_dimension_coordinates_expression(coords, exception):
        class _DerivedSeries(GenericSeries):
            _required_dimension_coordinates = dict(a=["x", "y"])

        expr = "a*x + b"
        parameters = {}
        for k, v in coords.items():
            parameters[k] = xr.DataArray(Q_(np.zeros(len(v))), dims=[k], coords={k: v})

        if exception is None:
            _DerivedSeries(expr, parameters=parameters)
            return

        with pytest.raises(exception):
            _DerivedSeries(expr, parameters=parameters)

    @staticmethod
    @pytest.mark.parametrize(
        "data, units, exception",
        [
            (Q_([[1, 2], [3, 4]], "m"), None, None),
            (Q_([[1, 2], [3, 4]], "s"), None, pint.DimensionalityError),
            ("a*b", dict(a="m"), None),
            ("a*b", dict(b="m"), None),
            ("a*b", dict(a="m/s", b="s"), None),
            ("a*b", dict(a="m/s", b="m"), pint.DimensionalityError),
            ("a*b", dict(a="s"), pint.DimensionalityError),
            ("a*b", None, pint.DimensionalityError),
        ],
    )
    def test_required_unit_dimensionality(data, units, exception):
        class _DerivedSeries(GenericSeries):
            _required_unit_dimensionality = U_("m")

        if exception is None:
            _DerivedSeries(data, units=units)
            return

        with pytest.raises(exception):
            _DerivedSeries(data, units=units)
