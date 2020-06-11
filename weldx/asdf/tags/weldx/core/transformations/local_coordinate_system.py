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
        """Convert a LocalCoordinateSystem to a tagged tree"""
        tree = {"dataset": node.dataset}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to a LocalCoordinateSystem"""
        dataset = tree["dataset"]
        return LocalCoordinateSystem(
            orientation=dataset.orientation.data, coordinates=dataset.coordinates.data
        )
