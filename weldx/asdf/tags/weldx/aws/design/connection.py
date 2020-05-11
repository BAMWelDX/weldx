from dataclasses import dataclass

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree

from .joint_penetration import JointPenetration
from .weld_details import WeldDetails

__all__ = ["Connection", "ConnectionType"]


@dataclass
class Connection:
    """<CLASS DOCSTRING>"""

    joint_type: str
    weld_type: str
    joint_penetration: JointPenetration
    weld_details: WeldDetails


class ConnectionType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/connection"
    version = "1.0.0"
    types = [Connection]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Connection, ctx):
        """convert to tagged tree and remove all None entries from node dictionary

        Parameters
        ----------
        node: Connection :

        ctx :


        Returns
        -------

        """
        tree = dict_to_tagged_tree(node, ctx)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Connection(**tree)
        return obj
