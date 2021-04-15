from dataclasses import dataclass, field
from typing import List
from uuid import UUID, uuid4

import networkx as nx

from weldx.asdf.types import WeldxType


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
        """Convert to python dict."""
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

    edges: List["DiEdge"] = field(default_factory=list)
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
        """Convert to python dict."""
        if not node.attributes:
            node.attributes = None
        return node.__dict__

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct from tree."""
        return DiNode(**tree)


# Graph --------------------------------------------------------------------------------


def build_tree(
    graph: nx.DiGraph, name: str, parent: str = None, keep_uuid: bool = False
) -> DiNode:
    """Recursively build a tree structure of the graph starting from node ``name``.

    Parameters
    ----------
    graph :
        Complete graph to build.
    name :
        Name (or key) of the current node.
    parent :
        Key of the node
    keep_uuid :
        store unique id node names

    Returns
    -------
    DiNode
        The root node object of the graph.

    """
    node = DiNode(name=name)

    for n in graph.successors(name):
        if not n == parent:
            child_node = build_tree(graph, n, parent=name, keep_uuid=keep_uuid)
            if child_node:
                edge = DiEdge(child_node, attributes=graph.edges[name, n])
                node.edges.append(edge)
    for n in graph.predecessors(name):
        if not n == parent:
            child_node = build_tree(graph, n, parent=name, keep_uuid=keep_uuid)
            if child_node:
                edge = DiEdge(
                    child_node, attributes=graph.edges[n, name], direction="bwd"
                )
                node.edges.append(edge)
    node.attributes = graph.nodes[name]  # add node attributes

    if isinstance(node.name, UUID) and not keep_uuid:
        node.name = None

    if not node.edges:
        node.edges = None

    return node


def build_graph(current_node: DiNode, graph: nx.DiGraph = None) -> nx.DiGraph:
    """Recursively rebuild a (partial) graph from a DiNode object.

    Parameters
    ----------
    current_node :
        The current DiNode to be added to the graph.
    graph :
        The graph object that is being built.

    Returns
    -------
    networkx.DiGraph
        The constructed graph.

    """
    if graph is None:
        graph = nx.DiGraph()

    name = current_node.name
    graph.add_node(name, **current_node.attributes)
    for edge in current_node.edges:
        attr = edge.attributes
        if edge.direction == "bwd":
            graph.add_edge(edge.target_node.name, name, **attr)
        else:
            graph.add_edge(name, edge.target_node.name, **attr)
        build_graph(edge.target_node, graph)
    return graph


class DiGraphTypeASDF(WeldxType):
    """Serialization class for `networkx.DiGraph`."""

    name = "core/graph/di_graph"
    version = "1.0.0"
    types = [nx.DiGraph]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: nx.DiGraph, ctx):
        """Check graph structure and build nested dictionary."""
        if not nx.is_tree(node):  # no cycles, single tree
            raise ValueError("Graph must represent a tree.")

        keep_uuid = getattr(node, "_wx_keep_uuid_name", False)

        root = build_tree(node, tuple(node.nodes)[0], keep_uuid=keep_uuid)
        return dict(root_node=root)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Rebuild directed graph from nested dictionary structure."""
        return build_graph(tree["root_node"])
