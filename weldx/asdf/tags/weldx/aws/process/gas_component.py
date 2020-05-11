from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree

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
        """convert to tagged tree and remove all None entries from node dictionary

        Parameters
        ----------
        node :

        ctx :


       """
        tree = dict_to_tagged_tree(node, ctx)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = GasComponent(**tree)
        return obj
