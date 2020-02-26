from dataclasses import dataclass
from typing import List  # noqa: F401
from asdf.yamlutil import custom_tree_to_tagged_tree
from weldx.asdf.types import WeldxType

import pint

__all__ = ["GasComponent", "GasComponentType"]


@dataclass
class GasComponent:
    """<CLASS DOCSTRING>"""

    gas_chemical_name: str
    gas_percentage: pint.Quantity


class GasComponentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/gas_component"
    version = "1.0.0"
    types = [GasComponent]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        # convert to tagged tree
        tree_full = dict(
            gas_chemical_name=custom_tree_to_tagged_tree(node.gas_chemical_name, ctx),
            gas_percentage=custom_tree_to_tagged_tree(node.gas_percentage, ctx),
        )

        # drop None values
        tree = {k: v for (k, v) in tree_full.items() if v is not None}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = GasComponent(**tree)
        return obj
