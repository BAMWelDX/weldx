from weldx.measurement import DataTransformation

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["DataTransformation", "DataTransformationType"]


class DataTransformationType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "measurement/data_transformation"
    version = "1.0.0"
    types = [DataTransformation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: DataTransformation, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = DataTransformation(**tree)
        return obj
