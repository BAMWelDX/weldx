import pint
from xarray import DataArray

from weldx.asdf.tags.weldx.core.common_types import Variable
from weldx.asdf.types import WeldxType
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
            tree = {
                "time": node.time,
                "data": node.data.magnitude,
                "unit": str(node.units),
            }
        else:
            tree = {"data": node.data, "unit": str(node.units)}

        return tree

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

        return None
