from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx

from weldx.asdf.types import WeldxType


# Signal -------------------------------------------------------------------------------
@dataclass
class SignalClass:
    """Doc."""

    name: str
    value: float = 3.14


class SignalClassTypeASDF(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

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
    """Doc."""

    func: str = "a*b"


class SignalTransformTypeASDF(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

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


# Edge ---------------------------------------------------------------------------------


@dataclass
class MeasurementChain:
    """Example Measurement Chain implementation."""

    graph: nx.DiGraph


class MeasurementChainTypeASDF(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/graph/measurement_chain"
    version = "1.0.0"
    types = [MeasurementChain]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: MeasurementChain, ctx):
        """convert to python dict"""
        root_node = build_node(node.graph, tuple(node.graph.nodes)[0])
        return dict(root_node=root_node)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        graph = nx.DiGraph()
        add_nodes(graph, tree["root_node"])
        return cls.types[0](graph=graph)


# Edge ---------------------------------------------------------------------------------
@dataclass
class Edge:
    """Generic graph edge type."""

    target_node: "Node"
    attributes: dict = field(default_factory=dict)
    direction: str = "fwd"


class EdgeTypeASDF(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/graph/edge"
    version = "1.0.0"
    types = [Edge]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: Edge, ctx):
        """convert to python dict"""
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return cls.types[0](**tree)


# Node ---------------------------------------------------------------------------------
@dataclass
class Node:
    """Generic graph node type."""

    name: str
    edges: List["Edge"]
    attributes: dict = field(default_factory=dict)


class NodeTypeASDF(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/graph/node"
    version = "1.0.0"
    types = [Node]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: Node, ctx):
        """convert to python dict"""
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return cls.types[0](**tree)


# Graph --------------------------------------------------------------------------------


def build_node(graph, name, parent=None):
    """Recursively build a tree structure starting from node ``name``."""
    node = Node(name=name, edges=[])
    for n in graph.neighbors(name):
        if not n == parent:
            child_node = build_node(graph, n, parent=name)
            edge = Edge(child_node, attributes=graph.edges[name, n])
            node.edges.append(edge)
    for n in graph.predecessors(name):
        if not n == parent:
            child_node = build_node(graph, n, parent=name)
            edge = Edge(child_node, attributes=graph.edges[n, name], direction="bwd")
            node.edges.append(edge)
    node.attributes = graph.nodes[name]  # add node attributes
    return node


def add_nodes(graph: nx.DiGraph, current_node: Node):
    """Doc."""
    name = current_node.name
    graph.add_node(name, **current_node.attributes)
    for edge in current_node.edges:
        attr = edge.attributes
        if edge.direction == "bwd":
            graph.add_edge(edge.target_node.name, name, **attr)
        else:
            graph.add_edge(name, edge.target_node.name, **attr)
        add_nodes(graph, edge.target_node)


class GraphTypeASDF(WeldxType):
    """Serialization class for `networkx.DiGraph`."""

    name = "core/graph/graph"
    version = "1.0.0"
    types = [nx.DiGraph]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: nx.DiGraph, ctx):
        """Doc."""
        root = build_node(node, tuple(node.nodes)[0])
        return dict(root_node=root)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Doc."""
        graph = nx.DiGraph()
        add_nodes(graph, tree["root_node"])
        return graph
