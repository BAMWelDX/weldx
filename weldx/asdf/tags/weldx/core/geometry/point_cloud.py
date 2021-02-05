from weldx.asdf.types import WeldxType
from weldx.geometry import PointCloud


class PointCloudTypeASDF(WeldxType):
    name = "core/geometry/point_cloud"
    version = "1.0.0"
    types = [PointCloud]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: PointCloud, ctx):
        """
        Convert an 'weldx.geometry.point_cloud' instance into YAML  representations.

        Parameters
        ----------
        node :
            Instance of the 'weldx.geometry.point_cloud' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the
            'weldx.core.ExternalFile' type to be serialized.

        """
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx) -> PointCloud:
        """
        Converts basic types representing YAML trees into an
        'weldx.geometry.point_cloud'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        weldx.geometry.point_cloud :
            An instance of the 'weldx.geometry.point_cloud' type.

        """
        return PointCloud(**tree)
