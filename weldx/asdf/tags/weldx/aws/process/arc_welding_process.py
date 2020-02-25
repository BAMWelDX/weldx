from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


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
        tree = dict(name=yamlutil.custom_tree_to_tagged_tree(node.name, ctx))
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ArcWeldingProcess(**tree)
        return obj
