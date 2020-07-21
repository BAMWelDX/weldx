"""Test the measurement package."""
from io import BytesIO

import asdf
import numpy as np
import pandas as pd
import pytest
import sympy
import xarray as xr

import weldx.measurement as msm
from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.core import MathematicalExpression


def test_generic_measurement():
    """Test basic measurement creation and ASDF read/write."""
    data_01 = msm.Data(
        name="Welding current", data=xr.DataArray([1, 2, 3, 4], dims=["time"])
    )

    data_02 = msm.Data(
        name="Welding voltage", data=xr.DataArray([10, 20, 30, 40], dims=["time"])
    )

    src_01 = msm.Source(
        name="Current Sensor",
        output_signal=msm.Signal("analog", "V", data=None),
        error=msm.Error(1337.42),
    )

    src_02 = msm.Source(
        name="Voltage Sensor",
        output_signal=msm.Signal("analog", "V", data=None),
        error=msm.Error(1),
    )

    dp_01 = msm.DataTransformation(
        name="AD conversion current measurement",
        input_signal=src_01.output_signal,
        output_signal=msm.Signal("digital", "V", data=None),
        error=msm.Error(999.0),
    )

    dp_02 = msm.DataTransformation(
        name="Calibration current measurement",
        input_signal=dp_01.output_signal,
        output_signal=msm.Signal("digital", "A", data=data_01),
        error=msm.Error(43.0),
    )

    dp_03 = msm.DataTransformation(
        name="AD conversion voltage measurement",
        input_signal=dp_02.output_signal,
        output_signal=msm.Signal("digital", "V", data=None),
        error=msm.Error(2.0),
    )

    dp_04 = msm.DataTransformation(
        name="Calibration voltage measurement",
        input_signal=dp_03.output_signal,
        output_signal=msm.Signal("digital", "V", data=data_02),
        error=msm.Error(Q_(3.0, "percent")),
    )

    chn_01 = msm.MeasurementChain(
        name="Current measurement", data_source=src_01, data_processors=[dp_01, dp_02]
    )

    chn_02 = msm.MeasurementChain(
        name="Voltage measurement", data_source=src_02, data_processors=[dp_03, dp_04]
    )

    eqp_01 = msm.GenericEquipment(
        "Current Sensor", sources=[src_01], data_transformations=[dp_02]
    )
    eqp_02 = msm.GenericEquipment(
        "AD Converter", sources=None, data_transformations=[dp_01, dp_03]
    )
    eqp_03 = msm.GenericEquipment(
        "Voltage Sensor", sources=None, data_transformations=[dp_04]
    )

    measurement_01 = msm.Measurement(
        name="Current measurement", data=[data_01], measurement_chain=chn_01
    )
    measurement_02 = msm.Measurement(
        name="Voltage measurement", data=[data_02], measurement_chain=chn_02
    )

    equipment = [eqp_01, eqp_02, eqp_03]
    measurement_data = [data_01, data_02]
    measurement_chains = [chn_01]
    measurements = [measurement_01, measurement_02]
    sources = [src_01]
    processors = [dp_01, dp_02]

    [a, x, b] = sympy.symbols("a x b")
    expr_01 = MathematicalExpression(a * x + b)
    expr_01.set_parameter("a", 2)
    expr_01.set_parameter("b", 3)
    print(expr_01.parameters)
    print(expr_01.get_variable_names())
    print(expr_01.evaluate(x=3))

    tree = {
        "equipment": equipment,
        "data": measurement_data,
        "measurements": measurements,
        # "expression": expr_01,
        "measurement_chains": measurement_chains,
        "data_sources": sources,
        "data_processors": processors,
    }

    asdf_buffer = BytesIO()

    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to(asdf_buffer)
        asdf_buffer.seek(0)

    asdf.open(asdf_buffer, extensions=[WeldxExtension(), WeldxAsdfExtension()])


# TimeSeries ---------------------------------------------------------------------------


def test_time_series_construction():
    # single value ----------------------------------------
    value = Q_(1, "m")
    ts_constant = msm.TimeSeries(data=value)

    assert ts_constant.data == value
    assert ts_constant.time is None
    assert ts_constant.interpolation == "linear"
    assert ts_constant.shape == tuple([1])
    assert value.check(UREG.get_dimensionality(ts_constant.units))

    # discrete values -------------------------------------
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = msm.TimeSeries(data=values, time=time, interpolation="step")

    assert np.all(ts_discrete.time == time)
    assert np.all(ts_discrete.data == values)
    assert ts_discrete.interpolation == "step"
    assert ts_discrete.shape == tuple([5])
    assert values.check(UREG.get_dimensionality(ts_discrete.units))

    # mathematical expression -----------------------------
    # scalar
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "1/s"), "b": Q_(-2, "")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = msm.TimeSeries(data=expr)

    assert ts_expr.time is None
    assert ts_expr.interpolation is None
    assert ts_expr.shape == tuple([1])

    assert isinstance(ts_expr.data, MathematicalExpression)
    assert ts_expr.data.num_variables() == 1
    assert ts_expr.data.num_parameters() == 2
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

    ts_expr_vec = msm.TimeSeries(data=expr_vec)

    assert ts_expr_vec.time is None
    assert ts_expr_vec.interpolation is None
    assert ts_expr_vec.shape == tuple([1, 3])

    assert isinstance(ts_expr_vec.data, MathematicalExpression)
    assert ts_expr_vec.data.num_variables() == 1
    assert ts_expr_vec.data.num_parameters() == 2
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
        msm.TimeSeries(values, time, "super interpolator 2000")
    with pytest.raises(ValueError):
        msm.TimeSeries(values, time, None)

    # too many free variables
    with pytest.raises(Exception):
        expr_2 = MathematicalExpression(expression=expr_string, parameters={})
        msm.TimeSeries(data=expr_2)
    # incompatible parameter units
    with pytest.raises(Exception):
        expr_3 = MathematicalExpression(
            expression=expr_string, parameters={"a": Q_(2, "1/s"), "b": Q_(-2, "m")}
        )
        msm.TimeSeries(data=expr_3)
    # cannot be evaluated with time vectors
    with pytest.raises(Exception):
        expr_4 = MathematicalExpression(
            expression=expr_string,
            parameters={"a": Q_([2, 3, 4], "1/s"), "b": Q_([-2, 3, 1], "")},
        )
        msm.TimeSeries(data=expr_4)


def test_time_series_interp_time_constant():
    value = Q_(1, "m")
    ts_constant = msm.TimeSeries(data=value)

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


def test_time_series_interp_time_discrete_linear():
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = msm.TimeSeries(data=values, time=time, interpolation="linear")

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
    # scalar ----------------------------------------------
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "meter/second"), "b": Q_(-2, "meter")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = msm.TimeSeries(data=expr)

    # single timedelta
    time_single = Q_(1, "second")
    value_interp_single = ts_expr.interp_time(time_single)

    assert value_interp_single == Q_(0, "meter")

    # multiple time deltas
    time_multi = Q_([0, 1, 2, 10], "second")
    value_interp_multi = ts_expr.interp_time(time_multi)

    assert len(value_interp_multi) == 4

    for i in range(4):
        assert (
            value_interp_multi[i] == parameters["a"] * time_multi[i] + parameters["b"]
        )

    # vector -----------------------------------------------
    expr_string_vec = "a*t+b"
    parameters_vec = {"a": Q_([[2, 3, 4]], "1/s"), "b": Q_([[-2, 3, 1]], "")}
    expr_vec = MathematicalExpression(
        expression=expr_string_vec, parameters=parameters_vec
    )

    ts_expr_vec = msm.TimeSeries(data=expr_vec)

    # single time delta
    value_interp_vec_single = ts_expr_vec.interp_time(time_single)

    assert np.all(np.isclose(value_interp_vec_single, [0, 6, 5]))

    # multiple time deltas
    value_interp_vec_multi = ts_expr_vec.interp_time(time_multi)

    assert value_interp_vec_multi.shape == tuple([4, 3])

    for i in range(4):
        assert (
            value_interp_multi[i] == parameters["a"] * time_multi[i] + parameters["b"]
        )

    # exceptions ------------------------------------------
    with pytest.raises(ValueError):
        ts_expr.interp_time(Q_(2, "s/m"))
