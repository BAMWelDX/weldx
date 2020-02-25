from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


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
        tree = dict(geometry=yamlutil.custom_tree_to_tagged_tree(node.geometry, ctx))
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Workpiece(**tree)
        return obj
