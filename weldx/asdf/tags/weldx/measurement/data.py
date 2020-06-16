from weldx.measurement import Data

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["Data", "DataType"]


class DataType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "measurement/data"
    version = "1.0.0"
    types = [Data]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Data, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Data(**tree)
        return obj
