from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf import yamlutil
from weldx.asdf.types import WeldxType
from .gas_component import GasComponent


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
            gas_component=yamlutil.custom_tree_to_tagged_tree(node.gas_component, ctx),
            common_name=yamlutil.custom_tree_to_tagged_tree(node.common_name, ctx),
            designation=yamlutil.custom_tree_to_tagged_tree(node.designation, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasType(**tree)
        return obj
