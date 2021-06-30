import pandas as pd
import pint

from weldx.asdf.tags.weldx.core.common_types import Variable
from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_shape_validator
from weldx.core import TimeSeries
from weldx.transformations import LocalCoordinateSystem


class LocalCoordinateSystemASDF(WeldxType):
    """Serialization class for weldx.transformations.LocalCoordinateSystem"""

    name = "core/transformations/local_coordinate_system"
    version = "1.0.0"
    types = [LocalCoordinateSystem]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {"wx_shape": wx_shape_validator}

    @classmethod
    def to_tree(cls, node: LocalCoordinateSystem, ctx):
        """
        Convert a 'LocalCoordinateSystem' instance into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'LocalCoordinateSystem' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'LocalCoordinateSystem'
            type to be serialized.

        """
        tree = {}

        orientations = None
        if not node.is_unity_rotation:
            orientations = Variable(
                "orientations", node.orientation.dims, node.orientation.data
            )
            if "time" not in node.orientation.coords:
                ctx.set_array_storage(orientations.data, "inline")
            tree["orientations"] = orientations

        coordinates = None
        if isinstance(node.coordinates, TimeSeries):
            tree["coordinates"] = node.coordinates
        elif not node.is_unity_translation:
            coordinates = Variable(
                "coordinates", node.coordinates.dims, node.coordinates.data
            )
            if "time" not in node.coordinates.coords:
                if isinstance(coordinates.data, pint.Quantity):
                    ctx.set_array_storage(coordinates.data.magnitude, "inline")
                else:
                    ctx.set_array_storage(coordinates.data, "inline")
            tree["coordinates"] = coordinates

        if "time" in node.dataset.coords:
            tree["time"] = pd.TimedeltaIndex(node.time)

        if node.reference_time is not None:
            tree["reference_time"] = node.reference_time

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into a 'LocalCoordinateSystem'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        LocalCoordinateSystem :
            An instance of the 'LocalCoordinateSystem' type.

        """
        orientations = tree.get("orientations")
        if orientations is not None:
            orientations = orientations.data

        coordinates = tree.get("coordinates")
        if coordinates is not None and not isinstance(coordinates, TimeSeries):
            coordinates = tree["coordinates"].data

        return LocalCoordinateSystem(
            orientation=orientations,
            coordinates=coordinates,
            time=tree.get("time"),
            time_ref=tree.get("reference_time"),
        )
