import asdf
import pytest

from weldx import Q_
from weldx.asdf.util import _write_buffer, _write_read_buffer
from weldx.core import MathematicalExpression
from weldx.measurement import (
    Error,
    MeasurementChain,
    Signal,
    SignalSource,
    SignalTransformation,
)


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_coordinate_system_manager(copy_arrays, lazy_load):
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

    tree = {"m_chain": mc}
    with asdf.AsdfFile(tree) as ff:
        ff.write_to("test.yaml")
    data = _write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    mc_file = data["m_chain"]
    # assert mc == mc_file
