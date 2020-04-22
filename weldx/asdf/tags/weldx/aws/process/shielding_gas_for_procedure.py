from dataclasses import dataclass
from typing import List  # noqa: F401

import pint
from asdf.yamlutil import custom_tree_to_tagged_tree

from weldx.asdf.types import WeldxType

from .shielding_gas_type import ShieldingGasType

__all__ = ["ShieldingGasForProcedure", "ShieldingGasForProcedureType"]


@dataclass
class ShieldingGasForProcedure:
    """<CLASS DOCSTRING>"""

    use_torch_shielding_gas: bool
    torch_shielding_gas: ShieldingGasType
    torch_shielding_gas_flowrate: pint.Quantity
    use_backing_gas: bool = None
    backing_gas: ShieldingGasType = None
    backing_gas_flowrate: pint.Quantity = None
    use_trailing_gas: bool = None
    trailing_shielding_gas: ShieldingGasType = None
    trailing_shielding_gas_flowrate: pint.Quantity = None


class ShieldingGasForProcedureType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/shielding_gas_for_procedure"
    version = "1.0.0"
    types = [ShieldingGasForProcedure]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            use_torch_shielding_gas=custom_tree_to_tagged_tree(
                node.use_torch_shielding_gas, ctx
            ),
            torch_shielding_gas=custom_tree_to_tagged_tree(
                node.torch_shielding_gas, ctx
            ),
            torch_shielding_gas_flowrate=custom_tree_to_tagged_tree(
                node.torch_shielding_gas_flowrate, ctx
            ),
            use_backing_gas=custom_tree_to_tagged_tree(node.use_backing_gas, ctx),
            backing_gas=custom_tree_to_tagged_tree(node.backing_gas, ctx),
            backing_gas_flowrate=custom_tree_to_tagged_tree(
                node.backing_gas_flowrate, ctx
            ),
            use_trailing_gas=custom_tree_to_tagged_tree(node.use_trailing_gas, ctx),
            trailing_shielding_gas=custom_tree_to_tagged_tree(
                node.trailing_shielding_gas, ctx
            ),
            trailing_shielding_gas_flowrate=custom_tree_to_tagged_tree(
                node.trailing_shielding_gas_flowrate, ctx
            ),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasForProcedure(**tree)
        return obj
