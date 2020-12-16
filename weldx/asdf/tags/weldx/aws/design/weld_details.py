from dataclasses import dataclass

import pint

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
    def to_tree(cls, node: WeldDetails, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = WeldDetails(**tree)
        return obj
