from dataclasses import dataclass
from typing import List  # noqa: F401

import pint
from asdf.yamlutil import custom_tree_to_tagged_tree

from weldx.asdf.types import WeldxType

__all__ = ["WeldDetails", "WeldDetailsType"]


@dataclass
class WeldDetails:
    """<CLASS DOCSTRING>"""

    joint_design: str
    weld_sizes: pint.Quantity
    number_of_passes: int


class WeldDetailsType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weld_details"
    version = "1.0.0"
    types = [WeldDetails]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            joint_design=custom_tree_to_tagged_tree(node.joint_design, ctx),
            weld_sizes=custom_tree_to_tagged_tree(node.weld_sizes, ctx),
            number_of_passes=custom_tree_to_tagged_tree(node.number_of_passes, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = WeldDetails(**tree)
        return obj
