import pint
import numpy as np

from weldx import Q_
from weldx.constants import WELDX_UNIT_REGISTRY
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove


@WELDX_UNIT_REGISTRY.check(None, "[length]", "[length]/[time]", "[length]")
# TODO: extend the ckeck decorator to check on output! # ret='[length]/[time]')
def compute_welding_speed(
    groove: IsoBaseGroove,
    seam_length: pint.Quantity,
    wire_feed: pint.Quantity,
    wire_diameter=Q_(4, "mm"),
):
    """Computes how fast the torch has to be moved with the given seam length and feed
    to fill the gap of the groove.

    Parameters
    ----------
    groove: IsoBaseGroove
        groove definition to compute welding speed for.
    seam_length: pint.Quantity["length"]
        length of the seam
    wire_feed: pint.Quantity["length/time"]
        feed of the wire
    wire_diameter: pint.Quantity["length"]
        diameter of welding wire

    Returns
    -------
    speed: pint.Quantity["length/time"]
    """
    groove_area = groove.cross_sect_area
    wire_area = np.pi / 4 * wire_diameter**2
    weld_speed = wire_area * wire_feed / groove_area * seam_length

    weld_speed.ito_reduced_units()
    return weld_speed
