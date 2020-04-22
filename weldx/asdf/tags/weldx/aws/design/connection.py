from dataclasses import dataclass
from typing import List  # noqa: F401

from asdf.yamlutil import custom_tree_to_tagged_tree

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
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            joint_type=custom_tree_to_tagged_tree(node.joint_type, ctx),
            weld_type=custom_tree_to_tagged_tree(node.weld_type, ctx),
            joint_penetration=custom_tree_to_tagged_tree(node.joint_penetration, ctx),
            weld_details=custom_tree_to_tagged_tree(node.weld_details, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Connection(**tree)
        return obj
