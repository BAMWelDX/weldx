"""Collection of welding utilities."""
import numpy as np

from weldx.constants import WELDX_UNIT_REGISTRY
from weldx.core import Q_, TimeSeries
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove


@WELDX_UNIT_REGISTRY.check(None, "[length]/[time]", "[length]")
def compute_welding_speed(
    groove: IsoBaseGroove,
    wire_feed: Q_,
    wire_diameter: Q_,
) -> TimeSeries:
    """Compute how fast the torch has to be moved to fill the given groove.

    Parameters
    ----------
    groove
        groove definition to compute welding speed for.
    wire_feed: pint.Quantity["length/time"]
        feed of the wire
    wire_diameter: pint.Quantity["length"]
        diameter of welding wire

    Returns
    -------
    speed: pint.Quantity["length/time"]
        The computed welding speed.

    """
    groove_area = groove.cross_sect_area
    wire_area = np.pi / 4 * wire_diameter ** 2
    weld_speed = wire_area * wire_feed / groove_area

    weld_speed.ito_reduced_units()
    return TimeSeries(weld_speed)
