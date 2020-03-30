from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf.yamlutil import custom_tree_to_tagged_tree
from weldx.asdf.types import WeldxType

__all__ = ["ArcWeldingProcess", "ArcWeldingProcessType"]


@dataclass
class ArcWeldingProcess:
    """<CLASS DOCSTRING>"""

    name: str


class ArcWeldingProcessType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/arc_welding_process"
    version = "1.0.0"
    types = [ArcWeldingProcess]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(name=custom_tree_to_tagged_tree(node.name, ctx))

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ArcWeldingProcess(**tree)
        return obj
