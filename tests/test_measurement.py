import asdf


import weldx.measurement as msm

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension


def test_generic():
    src_01 = msm.Source(
        name="Current Sensor", output_unit="V", error=msm.Error(1337.42)
    )

    dp_01 = msm.DataProcessor(
        name="AD converter", input_unit="V", output_unit="", error=msm.Error(999.0)
    )

    sources = [src_01]
    processors = [dp_01]

    tree = {"data_sources": sources, "data_processors": processors}
    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to("test.yaml")


# TODO: remove
test_generic()
