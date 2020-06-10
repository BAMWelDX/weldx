from dataclasses import dataclass

from weldx.asdf.types import WeldxType

__all__ = ["OneOfClass", "OneOfClassType"]


@dataclass
class OneOfClass:
    data: dict


class OneOfClassType(WeldxType):
    name = "debug/oneOf_class"
    version = "1.0.0"
    types = [OneOfClass]

    @classmethod
    def to_tree(cls, node: OneOfClass, ctx):
        return node.data

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = OneOfClass(tree)
        return obj
