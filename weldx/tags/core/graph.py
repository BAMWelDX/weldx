from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import networkx as nx

from weldx.asdf.types import WeldxConverter


# DiEdge -------------------------------------------------------------------------------
@dataclass
class DiEdge:
    """Generic directed edge type."""

    target_node: DiNode
    attributes: dict = field(default_factory=dict)
    direction: str = "fwd"


class DiEdgeConverter(WeldxConverter):
    """ASDF type for `DiEdge`."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/graph/di_edge-0.1.*"]
    types = [DiEdge]

    def to_yaml_tree(self, obj: DiEdge, tag: str, ctx) -> dict:
        """Convert to python dict."""
        if not obj.attributes:
            obj.attributes = None
        return obj.__dict__

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        return DiEdge(**node)


# DiNode -------------------------------------------------------------------------------
@dataclass
class DiNode:
    """Generic directed graph node type."""

    edges: list[DiEdge] = field(default_factory=list)
    name: str = field(default_factory=uuid4)
    attributes: dict = field(default_factory=dict)


class DiNodeConverter(WeldxConverter):
    """ASDF type for `DiNode`."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/graph/di_node-0.1.*"]
    types = [DiNode]

    def to_yaml_tree(self, obj: DiNode, tag: str, ctx) -> dict:
        """Convert to python dict."""
        if not obj.attributes:
            obj.attributes = None
        return obj.__dict__

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        return DiNode(**node)


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


class DiGraphConverter(WeldxConverter):
    """Serialization class for `networkx.DiGraph`."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/graph/di_graph-0.1.*"]
    types = [nx.DiGraph]

    def to_yaml_tree(self, obj: nx.DiGraph, tag: str, ctx) -> dict:
        """Check graph structure and build nested dictionary."""
        if not nx.is_tree(obj):  # no cycles, single tree
            raise ValueError("Graph must represent a tree.")

        keep_uuid = getattr(obj, "_wx_keep_uuid_name", False)

        root = build_tree(obj, tuple(obj.nodes)[0], keep_uuid=keep_uuid)
        return dict(root_node=root)

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Rebuild directed graph from nested dictionary structure."""
        return build_graph(node["root_node"])
