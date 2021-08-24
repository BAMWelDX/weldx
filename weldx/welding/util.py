"""Collection of welding utilities."""
from typing import Union

import numpy as np
import pandas as pd
import pint
import xarray as xr

from weldx.constants import Q_, WELDX_UNIT_REGISTRY
from weldx.core import MathematicalExpression, TimeSeries
from weldx.util import deprecated
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove

__all__ = ["compute_welding_speed"]


def sine(
    f: Union[pint.Quantity, str],
    amp: Union[pint.Quantity, str],
    bias: Union[pint.Quantity, str] = None,
    phase: Union[pint.Quantity, str] = Q_(0, "rad"),
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
    ts :

    """
    if bias is None:
        amp = Q_(amp)
        bias = 0.0 * amp.u
    expr_string = "a*sin(o*t+p)+b"
    parameters = {"a": amp, "b": bias, "o": Q_(2 * np.pi, "rad") * Q_(f), "p": phase}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)
    return TimeSeries(expr)


@deprecated(
    "0.4.1",
    "0.5.0",
    "The 'LocalCoordinateSystem' now supports 'TimeSeries' as coordinates rendering "
    "this function obsolete.",
)
def lcs_coords_from_ts(
    ts: TimeSeries, time: Union[pd.DatetimeIndex, pint.Quantity]
) -> xr.DataArray:
    """Create translation coordinates from a TimeSeries at specific timesteps.

    Parameters
    ----------
    ts:
        TimeSeries that describes the coordinate motion as a 3D vector.
    time
        Timestamps used for interpolation.
        TODO: add support for pd.DateTimeindex as well

    Returns
    -------
    xarray.DataArray :
        A DataArray with correctly labeled dimensions to be used for LCS creation.

    """
    ts_data = ts.interp_time(time=time).data_array
    # assign vector coordinates and convert to mm
    ts_data = ts_data.rename({"dim_1": "c"}).assign_coords({"c": ["x", "y", "z"]})
    ts_data.data = ts_data.data.to("mm").magnitude
    ts_data["time"] = pd.TimedeltaIndex(ts_data["time"].data)
    return ts_data


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
