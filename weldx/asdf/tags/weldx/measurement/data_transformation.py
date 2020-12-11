from weldx.asdf.types import WeldxType
from weldx.measurement import DataTransformation

__all__ = ["DataTransformation", "DataTransformationType"]


class DataTransformationType(WeldxType):
    """Serialization class for data transformations."""

    name = "measurement/data_transformation"
    version = "1.0.0"
    types = [DataTransformation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: DataTransformation, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = DataTransformation(**tree)
        return obj
