from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType

__all__ = ["JointPenetration", "JointPenetrationType"]


@dataclass
class JointPenetration:
    """<CLASS DOCSTRING>"""

    complete_or_partial: str
    root_penetration: pint.Quantity
    groove_weld_size: float = None
    incomplete_joint_penetration: float = None
    weld_size: float = None
    weld_size_E1: float = None
    weld_size_E2: float = None
    depth_of_fusion: float = None


class JointPenetrationType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/joint_penetration"
    version = "1.0.0"
    types = [JointPenetration]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: JointPenetration, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = JointPenetration(**tree)
        return obj
