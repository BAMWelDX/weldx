from dataclasses import dataclass
from typing import List  # noqa: F401

from asdf.yamlutil import custom_tree_to_tagged_tree

from weldx.asdf.types import WeldxType

from .sub_assembly import SubAssembly

__all__ = ["Weldment", "WeldmentType"]


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: List[SubAssembly]


class WeldmentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weldment"
    version = "1.0.0"
    types = [Weldment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            sub_assembly=custom_tree_to_tagged_tree(node.sub_assembly, ctx)
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Weldment(**tree)
        return obj
