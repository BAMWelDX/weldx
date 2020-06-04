from weldx.transformations import LocalCoordinateSystem
from weldx.asdf.types import WeldxType


class XarrayDataArrayASDF(WeldxType):
    """Serialization class for xarray.DataArray"""

    name = "core/transformations/local_coordinate_system"
    version = "1.0.0"
    types = [LocalCoordinateSystem]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: LocalCoordinateSystem, ctx):
        """Convert a LocalCoordinateSystem to a tagged tree"""
        tree = {"orientation": node.orientation, "coordinates": node.coordinates}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to a LocalCoordinateSystem"""
        return LocalCoordinateSystem(
            orientation=tree["orientation"], coordinates=tree["coordinates"]
        )
