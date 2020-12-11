from dataclasses import dataclass
from typing import List

from weldx.asdf.types import WeldxType

from .sub_assembly import SubAssembly

__all__ = ["Weldment", "WeldmentType"]


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: List[SubAssembly]


class WeldmentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weldment"
    version = "1.0.0"
    types = [Weldment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Weldment, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Weldment(**tree)
        return obj
