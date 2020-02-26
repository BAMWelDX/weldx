from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf.yamlutil import custom_tree_to_tagged_tree
from weldx.asdf.types import WeldxType

__all__ = ["Workpiece", "WorkpieceType"]


@dataclass
class Workpiece:
    """<CLASS DOCSTRING>"""

    geometry: str


class WorkpieceType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/workpiece"
    version = "1.0.0"
    types = [Workpiece]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(geometry=custom_tree_to_tagged_tree(node.geometry, ctx))

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Workpiece(**tree)
        return obj
