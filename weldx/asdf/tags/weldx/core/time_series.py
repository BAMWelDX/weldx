"""Contains the serialization class for the weldx.core.TimeSeries."""

import pint
from asdf.tags.core.ndarray import NDArrayType

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
            return {
                "time": node.time,
                "data": node.data.magnitude,
                "interpolation": node.interpolation,
                "unit": str(node.units),
            }
        return {"data": node.data, "unit": str(node.units)}

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
        if isinstance(tree["data"], NDArrayType):
            if "time" in tree:
                time = tree["time"]
            else:
                time = None
            NDArrayType.mro()
            values = Q_(tree["data"], tree["unit"])
            return TimeSeries(values, time, tree["interpolation"])

        return TimeSeries(tree["data"])
