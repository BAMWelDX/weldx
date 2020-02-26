from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf.yamlutil import custom_tree_to_tagged_tree
from weldx.asdf.types import WeldxType
from .gas_component import GasComponent

__all__ = ["ShieldingGasType", "ShieldingGasTypeType"]


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    gas_component: List[GasComponent]
    common_name: str
    designation: str = None


class ShieldingGasTypeType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/shielding_gas_type"
    version = "1.0.0"
    types = [ShieldingGasType]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            gas_component=custom_tree_to_tagged_tree(node.gas_component, ctx),
            common_name=custom_tree_to_tagged_tree(node.common_name, ctx),
            designation=custom_tree_to_tagged_tree(node.designation, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasType(**tree)
        return obj
