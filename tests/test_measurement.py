"""Test the measurement package."""
from io import BytesIO

import asdf
import numpy as np
import pandas as pd
import sympy
import xarray as xr

import weldx.measurement as msm
from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.tags.weldx.core.mathematical_expression import MathematicalExpression
from weldx.constants import WELDX_QUANTITY as Q_


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
    assert ts_constant.interpolation is None
    assert ts_constant.expression is None

    # discrete values -------------------------------------
    time = pd.TimedeltaIndex([0, 1, 2, 3, 4], unit="s")
    values = Q_(np.array([10, 11, 12, 14, 16]), "mm")
    ts_discrete = msm.TimeSeries(data=values, time=time, interpolation="constant")

    assert np.all(ts_discrete.time == time)
    assert np.all(ts_discrete.data == values)
    assert ts_discrete.interpolation == "constant"
    assert ts_discrete.expression is None

    # mathematical expression -----------------------------
    expr_string = "a*t+b"
    parameters = {"a": 2, "b": -2}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = msm.TimeSeries(data=expr)

    assert ts_expr.data is None
    assert ts_expr.time is None
    assert ts_expr.interpolation is None

    assert isinstance(ts_expr.expression, MathematicalExpression)
    assert ts_expr.expression.num_variables() == 1
    assert ts_expr.expression.num_parameters() == 2
    assert ts_expr.expression.get_variable_names()[0] == "t"

    for parameter in parameters:
        assert parameter in ts_expr.expression.parameters
        assert parameters[parameter] == ts_expr.expression.parameters[parameter]


def test_factories():
    pass


def test_time_series_interp_time_constant():
    value = Q_(1, "m")
    ts_constant = msm.TimeSeries(data=value)

    # single timedelta ------------------------------------
    time_delta_single = pd.TimedeltaIndex([2], "s")
    time_interp_single = ts_constant.interp_time(time_delta_single)

    assert time_interp_single == value

    # multiple time deltas --------------------------------
    time_delta_multi = pd.TimedeltaIndex([0, 2, 5], "s")
    value_interp_multi = ts_constant.interp_time(time_delta_multi)

    assert len(value_interp_multi) == 3
    for value_interp in value_interp_multi:
        assert value_interp == value


def test_time_series_interp_time_expression():
    expr_string = "a*t+b"
    parameters = {"a": Q_(2, "meter/second"), "b": Q_(-2, "meter")}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)

    ts_expr = msm.TimeSeries(data=expr)

    # single timedelta ------------------------------------
    time_single = Q_(1, "second")
    value_interp_single = ts_expr.interp_time(time_single)

    assert value_interp_single == Q_(0, "meter")

    # multiple time deltas --------------------------------
    time_multi = Q_([0, 1, 2, 10], "second")
    value_interp_multi = ts_expr.interp_time(time_multi)

    assert len(value_interp_multi) == 4

    for i in range(4):
        assert (
            value_interp_multi[i] == parameters["a"] * time_multi[i] + parameters["b"]
        )


# TODO: remove
test_time_series_interp_time_expression()
