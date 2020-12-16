from dataclasses import dataclass

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
    def to_tree(cls, node: Workpiece, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Workpiece(**tree)
        return obj
