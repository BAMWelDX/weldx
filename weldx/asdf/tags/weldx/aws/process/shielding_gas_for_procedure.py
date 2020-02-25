from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


@dataclass
class ShieldingGasForProcedure:
    """<CLASS DOCSTRING>"""

    use_torch_shielding_gas: bool
    torch_shielding_gas: GAS_TYPE
    torch_shielding_gas_flowrate: pint.Quantity
    use_backing_gas: bool
    backing_gas: GAS_TYPE
    backing_gas_flowrate: pint.Quantity
    use_trailing_gas: bool
    trailing_shielding_gas: GAS_TYPE
    trailing_shielding_gas_flowrate: pint.Quantity


class ShieldingGasForProcedureType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/shielding_gas_for_procedure"
    version = "1.0.0"
    types = [ShieldingGasForProcedure]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        tree = dict(
            use_torch_shielding_gas=yamlutil.custom_tree_to_tagged_tree(
                node.use_torch_shielding_gas, ctx
            ),
            torch_shielding_gas=yamlutil.custom_tree_to_tagged_tree(
                node.torch_shielding_gas, ctx
            ),
            torch_shielding_gas_flowrate=yamlutil.custom_tree_to_tagged_tree(
                node.torch_shielding_gas_flowrate, ctx
            ),
            use_backing_gas=yamlutil.custom_tree_to_tagged_tree(
                node.use_backing_gas, ctx
            ),
            backing_gas=yamlutil.custom_tree_to_tagged_tree(node.backing_gas, ctx),
            backing_gas_flowrate=yamlutil.custom_tree_to_tagged_tree(
                node.backing_gas_flowrate, ctx
            ),
            use_trailing_gas=yamlutil.custom_tree_to_tagged_tree(
                node.use_trailing_gas, ctx
            ),
            trailing_shielding_gas=yamlutil.custom_tree_to_tagged_tree(
                node.trailing_shielding_gas, ctx
            ),
            trailing_shielding_gas_flowrate=yamlutil.custom_tree_to_tagged_tree(
                node.trailing_shielding_gas_flowrate, ctx
            ),
        )
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasForProcedure(**tree)
        return obj
