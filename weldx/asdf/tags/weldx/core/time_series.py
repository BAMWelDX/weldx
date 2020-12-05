"""Contains the serialization class for the weldx.core.TimeSeries."""

import numpy as np
import pint

from weldx.asdf.types import WeldxType
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.core import TimeSeries


class TimeSeriesTypeASDF(WeldxType):
    """Serialization class for weldx.core.TimeSeries"""

    name = "core/time_series"
    version = "1.0.0"
    types = [TimeSeries]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: TimeSeries, ctx):
        """
        Convert an 'weldx.core.TimeSeries' instance into YAML  representations.

        Parameters
        ----------
        node :
            Instance of the 'weldx.core.TimeSeries' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'weldx.core.TimeSeries'
            type to be serialized.

        """

        if isinstance(node.data, pint.Quantity):
            if node.shape == tuple([1]):  # constant
                return {
                    "unit": str(node.units),
                    "value": node.data.magnitude[0],
                }
            else:
                return {
                    "time": node.time,
                    "unit": str(node.units),
                    "shape": node.shape,
                    "interpolation": node.interpolation,
                    "values": node.data.magnitude,
                }
        return {"expression": node.data, "unit": str(node.units), "shape": node.shape}

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into an 'weldx.core.TimeSeries'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        weldx.core.TimeSeries :
            An instance of the 'weldx.core.TimeSeries' type.

        """
        if "value" in tree:  # constant
            values = Q_(np.asarray(tree["value"]), tree["unit"])
            return TimeSeries(values)
        elif "values" in tree:
            time = tree["time"]
            interpolation = tree["interpolation"]
            values = Q_(tree["values"], tree["unit"])
            return TimeSeries(values, time, interpolation)

        return TimeSeries(tree["expression"])  # mathexpression
