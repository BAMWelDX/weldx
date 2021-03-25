from weldx.asdf.types import WeldxType
from weldx.measurement import Error, MeasurementChain, MeasurementChainGraph

__all__ = [
    "MeasurementChain",
    "MeasurementChainType",
    "MeasurementChainGraph",
    "MeasurementChainGraphType",
]


class MeasurementChainType(WeldxType):
    """Serialization class for measurement chains"""

    name = "measurement/measurement_chain"
    version = "1.0.0"
    types = [MeasurementChain]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: MeasurementChain, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = MeasurementChain(**tree)
        return obj


class MeasurementChainGraphType(WeldxType):
    """Serialization class for measurement chains"""

    name = "measurement/measurement_chain_graph"
    version = "1.0.0"
    types = [MeasurementChainGraph]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: MeasurementChainGraph, ctx):

        signals = [{"source": name, **attr} for name, attr in node._graph.nodes.items()]
        return {"name": node._name, "source": node._source, "signals": signals}

    @classmethod
    def from_tree(cls, tree, ctx):
        return MeasurementChainGraph.construct_from_tree(tree)
