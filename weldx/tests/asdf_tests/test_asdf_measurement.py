import pytest

from weldx import Q_
from weldx.asdf.util import _write_read_buffer
from weldx.core import MathematicalExpression
from weldx.measurement import (
    Error,
    GenericEquipment,
    MeasurementChain,
    Signal,
    SignalSource,
    SignalTransformation,
)


def measurement_chain_without_equipment() -> MeasurementChain:
    mc = MeasurementChain(
        "Current measurement chain",
        SignalSource(
            "Current measurement",
            Signal("analog", "V"),
            Error(Q_(0.1, "percent")),
        ),
    )
    mc.add_transformation(
        SignalTransformation(
            name="AD conversion",
            error=Error(Q_(0.5, "percent")),
            type_transformation="AD",
        )
    )
    mc.add_transformation(
        SignalTransformation(
            name="Calibration",
            error=Error(Q_(1.5, "percent")),
            func=MathematicalExpression(
                "a*x+b", parameters=dict(a=Q_(3, "A/V"), b=Q_(2, "A"))
            ),
        )
    )

    return mc


def measurement_chain_with_equipment() -> MeasurementChain:
    source = SignalSource(
        "Current measurement",
        output_signal=Signal(signal_type="analog", unit="V"),
        error=Error(Q_(1, "percent")),
    )
    ad_conversion = SignalTransformation(
        "AD conversion current measurement",
        error=Error(Q_(0, "percent")),
        func=MathematicalExpression(
            expression="a*x+b", parameters=dict(a=Q_(1, "1/V"), b=Q_(1, ""))
        ),
    )
    calibration = SignalTransformation(
        "Current measurement calibration",
        error=Error(Q_(1.2, "percent")),
        func=MathematicalExpression(
            expression="a*x+b", parameters=dict(a=Q_(1, "A"), b=Q_(1, "A"))
        ),
    )
    eq_source = GenericEquipment(
        name="Source Equipment",
        sources=[source],
    )
    eq_ad_conversion = GenericEquipment(
        name="AD Equipment", data_transformations=[ad_conversion]
    )
    eq_calibration = GenericEquipment(
        name="Calibration Equipment", data_transformations=[calibration]
    )
    mc = MeasurementChain.from_equipment("Measurement chain", eq_source)
    mc.add_transformation_from_equipment(eq_ad_conversion)
    mc.add_transformation_from_equipment(eq_calibration)
    return mc


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize(
    "measurement_chain",
    [measurement_chain_without_equipment(), measurement_chain_with_equipment()],
)
def test_coordinate_system_manager(copy_arrays, lazy_load, measurement_chain):
    tree = {"m_chain": measurement_chain}
    # todo: remove
    # with asdf.AsdfFile(tree) as ff:
    #    ff.write_to("test.yaml")
    data = _write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    mc_file = data["m_chain"]
    assert measurement_chain == mc_file
