from dataclasses import dataclass
from typing import List

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

from .connection import Connection
from .workpiece import Workpiece

__all__ = ["SubAssembly", "SubAssemblyType"]


@dataclass
class SubAssembly:
    """<CLASS DOCSTRING>"""

    workpiece: List[Workpiece]
    connection: Connection


class SubAssemblyType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/sub_assembly"
    version = "1.0.0"
    types = [SubAssembly]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: SubAssembly, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = SubAssembly(**tree)
        return obj
