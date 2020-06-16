from weldx.transformations import LocalCoordinateSystem
from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_shape_validator


class LocalCoordinateSystemASDF(WeldxType):
    """Serialization class for weldx.transformations.LocalCoordinateSystem"""

    name = "core/transformations/local_coordinate_system"
    version = "1.0.0"
    types = [LocalCoordinateSystem]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {}

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
        tree = {"dataset": node.dataset}
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
        dataset = tree["dataset"]
        return LocalCoordinateSystem(
            orientation=dataset.orientation.data, coordinates=dataset.coordinates.data
        )
