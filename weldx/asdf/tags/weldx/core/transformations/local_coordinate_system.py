from weldx.asdf.types import WeldxType
from weldx.transformations import LocalCoordinateSystem


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
        # example code to manipulate inline array storage
        if "time" not in node.coordinates.coords:
            ctx.set_array_storage(node.coordinates.data, "inline")
        if "time" not in node.orientation.coords:
            ctx.set_array_storage(node.orientation.data, "inline")
        ctx.set_array_storage(node.dataset.coords["c"].data, "inline")  # not working
        ctx.set_array_storage(
            node.orientation.coords["v"].data, "inline"
        )  # not working
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
