from dataclasses import dataclass
from typing import List  # noqa: F401

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["ConstClass", "ConstClassType"]


@dataclass
class ConstClass:
    """<TODO CLASS DOCSTRING>"""

    name: str
    value: float


class ConstClassType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "debug/const_class"
    version = "1.0.0"
    types = [ConstClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: ConstClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ConstClass(**tree)
        return obj
