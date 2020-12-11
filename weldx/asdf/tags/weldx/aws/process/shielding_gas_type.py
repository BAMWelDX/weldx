from dataclasses import dataclass
from typing import List

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
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShieldingGasType(**tree)
        return obj
