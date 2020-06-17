from weldx.measurement import MeasurementChain

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["MeasurementChain", "MeasurementChainType"]


class MeasurementChainType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "measurement/measurement_chain"
    version = "1.0.0"
    types = [MeasurementChain]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: MeasurementChain, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = MeasurementChain(**tree)
        return obj
