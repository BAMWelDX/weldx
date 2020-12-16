from dataclasses import dataclass

from weldx.asdf.types import WeldxType

from .joint_penetration import JointPenetration
from .weld_details import WeldDetails

__all__ = ["Connection", "ConnectionType"]


@dataclass
class Connection:
    """<CLASS DOCSTRING>"""

    joint_type: str
    weld_type: str
    joint_penetration: JointPenetration
    weld_details: WeldDetails


class ConnectionType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/connection"
    version = "1.0.0"
    types = [Connection]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Connection, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Connection(**tree)
        return obj
