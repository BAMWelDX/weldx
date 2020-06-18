import asdf
import xarray as xr

import weldx.measurement as msm

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension


def test_generic_save():
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
        input_signal=msm.Signal("analog", "V", data=None),
        output_signal=msm.Signal("digital", "V", data=None),
        error=msm.Error(999.0),
    )

    dp_02 = msm.DataTransformation(
        name="Calibration current measurement",
        input_signal=msm.Signal("digital", "V", data=None),
        output_signal=msm.Signal("digital", "A", data=data_01),
        error=msm.Error(43.0),
    )

    dp_03 = msm.DataTransformation(
        name="AD conversion voltage measurement",
        input_signal=msm.Signal("analog", "V", data=None),
        output_signal=msm.Signal("digital", "V", data=None),
        error=msm.Error(2.0),
    )

    dp_04 = msm.DataTransformation(
        name="Calibration voltage measurement",
        input_signal=msm.Signal("digital", "V", data=None),
        output_signal=msm.Signal("digital", "V", data=data_02),
        error=msm.Error(3.0),
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

    tree = {
        "equipment": equipment,
        "data": measurement_data,
        "measurements": measurements,
        # "measurement_chains": measurement_chains,
        # "data_sources": sources,
        # "data_processors": processors,
    }
    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to("test.yaml")


# TODO: remove
test_generic_save()


def test_generic_load():
    f = asdf.open("test.yaml", extensions=[WeldxExtension(), WeldxAsdfExtension()])

    # processors = f.tree["data_processors"]
    # sources = f.tree["data_sources"]
    # measurement_chains = f.tree["measurement_chains"]

    # print(processors[0])
    # print(sources[0])
    # print(measurement_chains[0])


# TODO: remove
test_generic_load()
