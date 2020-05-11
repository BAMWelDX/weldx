from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree

__all__ = ["WeldDetails", "WeldDetailsType"]


@dataclass
class WeldDetails:
    """<CLASS DOCSTRING>"""

    joint_design: str
    weld_sizes: pint.Quantity
    number_of_passes: int


class WeldDetailsType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weld_details"
    version = "1.0.0"
    types = [WeldDetails]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: WeldDetails, ctx):
        """convert to tagged tree and remove all None entries from node dictionary

        Parameters
        ----------
        node: WeldDetails :
            
        ctx :
            

        Returns
        -------

        """
        tree = dict_to_tagged_tree(node, ctx)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = WeldDetails(**tree)
        return obj
