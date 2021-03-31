"""Test welding util functions."""
import pint
import pytest

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.welding.groove.iso_9692_1 import IGroove
from weldx.welding.util import compute_welding_speed


def test_welding_speed():  # noqa
    groove = IGroove(b=Q_(10, "mm"), t=Q_(5, "cm"))
    wire_diameter = Q_(1, "mm")
    wire_feed = Q_(1, "mm/s")
    result = compute_welding_speed(groove, wire_feed, wire_diameter)
    assert result.units == wire_feed.units
    assert result > 0


def test_illegal_input_dimension():  # noqa
    groove = IGroove(b=Q_(10, "mm"), t=Q_(5, "cm"))
    with pytest.raises(pint.errors.DimensionalityError):
        # only a length for feed
        compute_welding_speed(groove, wire_feed=Q_(1, "mm"), wire_diameter=Q_(1, "mm"))

    with pytest.raises(pint.errors.DimensionalityError):
        # diameter wrong dimension
        compute_welding_speed(groove, wire_feed=Q_(1, "mm/s"), wire_diameter=Q_(1, "s"))
