"""Collection of welding utilities."""
from typing import Union

import numpy as np
import pint

from weldx import Q_
from weldx.constants import WELDX_UNIT_REGISTRY
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove

__all__ = ["compute_welding_speed"]


@WELDX_UNIT_REGISTRY.check(None, "[length]/[time]", "[length]")
def compute_welding_speed(
    groove: IsoBaseGroove,
    wire_feed: Union[pint.Quantity, str],
    wire_diameter: Union[pint.Quantity, str],
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
