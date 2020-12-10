from weldx.asdf.types import WeldxType
from weldx.measurement import Data

__all__ = ["Data", "DataType"]


class DataType(WeldxType):
    """Serialization class measurement data."""

    name = "measurement/data"
    version = "1.0.0"
    types = [Data]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Data, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Data(**tree)
        return obj
