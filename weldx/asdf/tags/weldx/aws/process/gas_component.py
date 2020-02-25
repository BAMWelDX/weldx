from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


@dataclass
class GasComponent:
    """<CLASS DOCSTRING>"""

    gas_chemical_name: str
    gas_percentage: float


class GasComponentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/gas_component"
    version = "1.0.0"
    types = [GasComponent]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        tree = dict(
            gas_chemical_name=yamlutil.custom_tree_to_tagged_tree(
                node.gas_chemical_name, ctx
            ),
            gas_percentage=yamlutil.custom_tree_to_tagged_tree(
                node.gas_percentage, ctx
            ),
        )
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = GasComponent(**tree)
        return obj
