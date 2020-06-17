import asdf


import weldx.measurement as msm

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension


def test_generic_save():

    src_01 = msm.Source(
        name="Current Sensor",
        output_signal_type="analog",
        output_unit="V",
        error=msm.Error(1337.42),
    )

    dp_01 = msm.DataProcessor(
        name="AD converter",
        input_signal_type="analog",
        input_unit="V",
        output_signal_type="digital",
        output_unit="V",
        error=msm.Error(999.0),
    )
    dp_02 = msm.DataProcessor(
        name="Current Sensor Calibration",
        input_signal_type="digital",
        input_unit="V",
        output_signal_type="digital",
        output_unit="A",
        error=msm.Error(43.0),
    )

    chn_01 = msm.MeasurementChain(
        name="Current measurement", source=src_01, data_processors=[dp_01, dp_02]
    )

    measurement_chains = [chn_01]
    sources = [src_01]
    processors = [dp_01, dp_02]

    tree = {
        "measurement_chains": measurement_chains,
        "data_sources": sources,
        "data_processors": processors,
    }
    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to("test.yaml")


# TODO: remove
test_generic_save()


def test_generic_load():
    f = asdf.open("test.yaml", extensions=[WeldxExtension(), WeldxAsdfExtension()])

    processors = f.tree["data_processors"]
    sources = f.tree["data_sources"]
    measurement_chains = f.tree["measurement_chains"]

    print(processors[0])
    print(sources[0])
    print(measurement_chains[0])


# TODO: remove
test_generic_load()
