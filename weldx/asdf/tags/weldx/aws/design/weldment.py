from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: list


class WeldmentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weldment"
    version = "1.0.0"
    types = [Weldment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        tree = dict(
            sub_assembly=yamlutil.custom_tree_to_tagged_tree(node.sub_assembly, ctx)
        )
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Weldment(**tree)
        return obj
