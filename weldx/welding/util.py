"""Collection of welding utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pint

from weldx.constants import Q_, WELDX_UNIT_REGISTRY
from weldx.core import MathematicalExpression, TimeSeries
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove

if TYPE_CHECKING:  # pragma: no cover
    from weldx.types import QuantityLike

__all__ = ["compute_welding_speed"]


def sine(
    f: QuantityLike,
    amp: QuantityLike,
    bias: QuantityLike = None,
    phase: QuantityLike = None,
) -> TimeSeries:
    """Create a simple sine TimeSeries from quantity parameters.

    f(t) = amp*sin(f*t+phase)+bias

    Parameters
    ----------
    f :
        Frequency of the sine (in Hz)
    amp :
        Sine amplitude
    bias :
        function bias
    phase :
        phase shift

    Returns
    -------
    weldx.TimeSeries :

    """
    if phase is None:
        phase = Q_(0, "rad")

    if bias is None:
        amp = Q_(amp)
        bias = 0.0 * amp.u
    expr_string = "a*sin(o*t+p)+b"
    parameters = {"a": amp, "b": bias, "o": Q_(2 * np.pi, "rad") * Q_(f), "p": phase}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)
    return TimeSeries(expr)


@WELDX_UNIT_REGISTRY.check(None, "[length]/[time]", "[length]")
def compute_welding_speed(
    groove: IsoBaseGroove,
    wire_feed: QuantityLike,
    wire_diameter: QuantityLike,
) -> pint.Quantity:
    """Compute how fast the torch has to be moved to fill the given groove.

    Parameters
    ----------
    groove
        groove definition to compute welding speed for.
    wire_feed: pint.Quantity
        feed of the wire, given in dimensionality "length/time".
    wire_diameter: pint.Quantity
        diameter of welding wire, given in dimensionality "length".

    Returns
    -------
    speed: pint.Quantity
        The computed welding speed, given in dimensionality "length/time".

    """
    groove_area = groove.cross_sect_area
    wire_area = np.pi / 4 * Q_(wire_diameter) ** 2
    weld_speed = wire_area * Q_(wire_feed) / groove_area

    weld_speed.ito_reduced_units()
    return weld_speed
