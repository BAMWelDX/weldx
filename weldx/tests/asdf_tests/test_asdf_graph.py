"""Test graph serializations."""
import unittest

import networkx as nx

from weldx.asdf.util import write_read_buffer

# --------------------------------------------------------------------------------------
# DiGraph
# --------------------------------------------------------------------------------------


class TestDiGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(
            [("A", "B"), ("A", "C"), ("A", "F"), ("D", "C"), ("B", "H"), ("X", "A")]
        )
        nx.set_node_attributes(self.graph, 3.14, "node_attr")
        nx.set_edge_attributes(self.graph, 42, "edge_attr")

    def test_graph_roundtrip(self):
        g = self.graph
        data = write_read_buffer(dict(graph=g))
        g2 = data["graph"]

        assert sorted(g2.edges) == sorted(g.edges)
        assert sorted(g2.nodes) == sorted(g.nodes)

        for node in g:
            assert g.nodes[node] == g2.nodes[node]

        for edge in g.edges:
            assert g.edges[edge] == g2.edges[edge]
