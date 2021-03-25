from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from weldx.asdf.types import WeldxType


@dataclass
class Edge:
    """Doc."""

    target_node: "Node"
    inverted: bool = None


@dataclass
class Node:
    """Doc."""

    name: str
    edges: List["Edge"]


class GraphTypeASDF(WeldxType):
    """Serialization class for `networkx.DiGraph`."""

    name = "core/graph/graph"
    version = "1.0.0"
    types = [nx.DiGraph]
    requires = ["weldx"]

    @classmethod
    def build_node(cls, graph, name, parent=None):
        """Recursively build a tree structure starting from nore ``name``."""
        node = Node(name=name, edges=[])
        for n in graph.neighbors(name):
            if not n == parent:
                child_node = cls.build_node(graph, n, parent=name)
                node.edges.append(Edge(child_node).__dict__)
        for n in graph.predecessors(name):
            if not n == parent:
                child_node = cls.build_node(graph, n, parent=name)
                node.edges.append(Edge(child_node, inverted=True).__dict__)
        return node.__dict__

    @classmethod
    def add_nodes(cls, graph: nx.DiGraph, current_node: Dict[str, List]):
        for edge in current_node["edges"]:
            if edge.get("inverted", None):
                graph.add_edge(edge["target_node"]["name"], current_node["name"])
            else:
                graph.add_edge(current_node["name"], edge["target_node"]["name"])
            cls.add_nodes(graph, edge["target_node"])

    @classmethod
    def to_tree(cls, node: nx.DiGraph, ctx):
        root = cls.build_node(node, tuple(node.nodes)[0])
        return dict(root_node=root)

    @classmethod
    def from_tree(cls, tree, ctx):
        graph = nx.DiGraph()
        cls.add_nodes(graph, tree["root_node"])
        return graph
