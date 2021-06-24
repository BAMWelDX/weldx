"""Test GMAW welding process schema implementation."""

import pytest

from weldx import Q_
from weldx.asdf.util import write_read_buffer
from weldx.core import TimeSeries
from weldx.welding.processes import GmawProcess


@pytest.mark.parametrize(
    "inputs",
    [
        GmawProcess(
            "spray",
            "Fronius",
            "TPSi",
            dict(
                wire_feedrate=Q_(10, "m/min"),
                voltage=TimeSeries(Q_(40.0, "V")),
                impedance=Q_(10.0, "percent"),
                characteristic=Q_(5, "V/A"),
            ),
        ),
        GmawProcess(
            "spray",
            "CLOOS",
            "Quinto",
            dict(
                wire_feedrate=Q_(10, "m/min"),
                voltage=TimeSeries(Q_([40.0, 20.0], "V"), Q_([0.0, 10.0], "s")),
                impedance=Q_(10.0, "percent"),
                characteristic=Q_(5, "V/A"),
            ),
            tag="CLOOS/spray_arc",
        ),
        GmawProcess(
            "pulse",
            "CLOOS",
            "Quinto",
            dict(
                wire_feedrate=Q_(10.0, "m/min"),
                pulse_voltage=Q_(40.0, "V"),
                pulse_duration=Q_(5.0, "ms"),
                pulse_frequency=Q_(100.0, "Hz"),
                base_current=Q_(60.0, "A"),
            ),
            tag="CLOOS/pulse",
            meta={"modulation": "UI"},
        ),
        GmawProcess(
            "pulse",
            "CLOOS",
            "Quinto",
            dict(
                wire_feedrate=Q_(10.0, "m/min"),
                pulse_current=Q_(0.3, "kA"),
                pulse_duration=Q_(5.0, "ms"),
                pulse_frequency=Q_(100.0, "Hz"),
                base_current=Q_(60.0, "A"),
            ),
            tag="CLOOS/pulse",
            meta={"modulation": "II"},
        ),
    ],
)
def test_gmaw_process(inputs):
    data = write_read_buffer({"root": inputs})
    assert data["root"] == inputs
