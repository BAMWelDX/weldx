from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType
from .gas_component import GasComponent


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    gas_component: List[GasComponent]
    common_name: str
    designation: str


class ShieldingGasTypeType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/shielding_gas_type"
    version = "1.0.0"
    types = [ShieldingGasType]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        tree = dict(
            gas_component=yamlutil.custom_tree_to_tagged_tree(node.gas_component, ctx),
            common_name=yamlutil.custom_tree_to_tagged_tree(node.common_name, ctx),
            designation=yamlutil.custom_tree_to_tagged_tree(node.designation, ctx),
        )
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasType(**tree)
        return obj
