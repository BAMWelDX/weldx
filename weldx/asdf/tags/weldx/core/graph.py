from dataclasses import dataclass, field
from typing import Dict, List
from uuid import uuid4

import networkx as nx

from weldx.asdf.types import WeldxType


# Signal -------------------------------------------------------------------------------
@dataclass
class SignalClass(object):
    """Dummy class representing a signal."""

    name: str
    value: float = 3.14


class SignalClassTypeASDF(WeldxType):
    """ASDF type for dummy signal"""

    name = "core/graph/signal"
    version = "1.0.0"
    types = [SignalClass]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: SignalClass, ctx):
        """convert to python dict"""
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return cls.types[0](**tree)


# SignalTransform ----------------------------------------------------------------------
@dataclass
class SignalTransform:
    """Dummy class representing a transformation between signals."""

    func: str = "a*b"


class SignalTransformTypeASDF(WeldxType):
    """ASDF type for dummy signal transform."""

    name = "core/graph/signal_transform"
    version = "1.0.0"
    types = [SignalTransform]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: SignalTransform, ctx):
        """convert to python dict"""
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return cls.types[0](**tree)


# MeasurementChain ---------------------------------------------------------------------


@dataclass
class MeasurementChain:
    """Example Measurement Chain implementation."""

    graph: nx.DiGraph


class MeasurementChainTypeASDF(WeldxType):
    """ASDF type for dummy Measurement Chain."""

    name = "core/graph/measurement_chain"
    version = "1.0.0"
    types = [MeasurementChain]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: MeasurementChain, ctx):
        """convert to python dict"""
        root_node = build_tree(node.graph, tuple(node.graph.nodes)[0])  # set root node
        return dict(root_node=root_node)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        graph = nx.DiGraph()
        build_graph(graph, tree["root_node"])
        return MeasurementChain(graph=graph)


# DiEdge -------------------------------------------------------------------------------
@dataclass
class DiEdge:
    """Generic directed edge type."""

    target_node: "DiNode"
    attributes: dict = field(default_factory=dict)
    direction: str = "fwd"


class DiEdgeTypeASDF(WeldxType):
    """ASDF type for `DiEdge`."""

    name = "core/graph/di_edge"
    version = "1.0.0"
    types = [DiEdge]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: DiEdge, ctx):
        """convert to python dict"""
        if not node.attributes:
            node.attributes = None
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return DiEdge(**tree)


# DiNode -------------------------------------------------------------------------------
@dataclass
class DiNode:
    """Generic directed graph node type."""

    edges: List["DiEdge"]
    name: str = field(default_factory=uuid4)
    attributes: dict = field(default_factory=dict)


class DiNodeTypeASDF(WeldxType):
    """ASDF type for `DiNode`."""

    name = "core/graph/di_node"
    version = "1.0.0"
    types = [DiNode]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: DiNode, ctx):
        """convert to python dict"""
        if not node.attributes:
            node.attributes = None
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return DiNode(**tree)


# Graph --------------------------------------------------------------------------------


def build_tree(
    graph: nx.DiGraph, name: str, parent: str = None, nodes: Dict[str, DiNode] = None
):
    """Recursively build a tree structure of the graph starting from node ``name``.

    Parameters
    ----------
    graph :
        Complete graph to build.
    name :
        Name (or key) of the current node.
    parent :
        Key of the node
    nodes:
        List of nodes already generated

    Returns
    -------
    DiNode
        The root node object of the graph.

    """
    if nodes is None:
        nodes = {}

    if node := nodes.get(name, None):
        if any(e.target_node.name in nodes for e in node.edges):
            return None
        return node

    node = DiNode(name=name, edges=[])
    nodes[name] = node

    for n in graph.neighbors(name):
        if not n == parent:
            child_node = build_tree(graph, n, parent=name, nodes=nodes)
            if child_node:
                edge = DiEdge(child_node, attributes=graph.edges[name, n])
                node.edges.append(edge)
    for n in graph.predecessors(name):
        if not n == parent:
            child_node = build_tree(graph, n, parent=name, nodes=nodes)
            if child_node:
                edge = DiEdge(
                    child_node, attributes=graph.edges[n, name], direction="bwd"
                )
                node.edges.append(edge)
    node.attributes = graph.nodes[name]  # add node attributes
    return node


def build_graph(
    graph: nx.DiGraph, current_node: DiNode, nodes: Dict[str, DiNode] = None, ctx=None
):
    """Recursively rebuild a (partial) graph from a DiNode object.

    Parameters
    ----------
    graph :
        The graph object that is being built.
    current_node :
        The current DiNode to be added to the graph.

    Returns
    -------

    """
    if nodes is None:
        nodes = {}

    name = current_node.name
    graph.add_node(name, **current_node.attributes)
    for edge in current_node.edges:
        attr = edge.attributes
        if edge.direction == "bwd":
            graph.add_edge(edge.target_node.name, name, **attr)
        else:
            graph.add_edge(name, edge.target_node.name, **attr)
        build_graph(graph, edge.target_node, nodes=nodes, ctx=ctx)


class DiGraphTypeASDF(WeldxType):
    """Serialization class for `networkx.DiGraph`."""

    name = "core/graph/di_graph"
    version = "1.0.0"
    types = [nx.DiGraph]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: nx.DiGraph, ctx):
        """Doc."""
        # if not nx.is_tree(node):  # no cycles, single tree
        #     raise ValueError("Graph must represent a tree.")

        root = build_tree(node, tuple(node.nodes)[0])
        return dict(root_node=root)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Doc."""
        graph = nx.DiGraph()
        build_graph(graph, tree["root_node"], ctx=ctx)
        return graph
