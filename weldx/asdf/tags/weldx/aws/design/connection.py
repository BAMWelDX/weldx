from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf import yamlutil
from weldx.asdf.types import WeldxType


@dataclass
class Connection:
    """<CLASS DOCSTRING>"""

    joint_type: str
    weld_type: str
    joint_penetration: joint_penetration
    weld_details: weld_details


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
            joint_type=yamlutil.custom_tree_to_tagged_tree(node.joint_type, ctx),
            weld_type=yamlutil.custom_tree_to_tagged_tree(node.weld_type, ctx),
            joint_penetration=yamlutil.custom_tree_to_tagged_tree(
                node.joint_penetration, ctx
            ),
            weld_details=yamlutil.custom_tree_to_tagged_tree(node.weld_details, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Connection(**tree)
        return obj
