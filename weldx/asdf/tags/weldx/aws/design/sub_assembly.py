from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf.yamlutil import custom_tree_to_tagged_tree
from weldx.asdf.types import WeldxType

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
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            workpiece=custom_tree_to_tagged_tree(node.workpiece, ctx),
            connection=custom_tree_to_tagged_tree(node.connection, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = SubAssembly(**tree)
        return obj
