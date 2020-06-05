from dataclasses import dataclass

from asdf import ValidationError

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["AnyOfClass", "AnyOfClassType"]


@dataclass
class AnyOfClass:
    data: dict


class AnyOfClassType(WeldxType):
    name = "debug/anyof_class"
    version = "1.0.0"
    types = [AnyOfClass]

    @classmethod
    def to_tree(cls, node: AnyOfClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        # tree = drop_none_attr(node)
        return node.data

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = AnyOfClass(tree)
        return obj
