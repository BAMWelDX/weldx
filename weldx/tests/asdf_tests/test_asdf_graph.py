"""Test graph serializations."""
import unittest

import networkx as nx
import pytest

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

    def test_graph_exceptions(self):
        self.graph.remove_edge("A", "C")  # two trees in graph
        with pytest.raises(ValueError):
            write_read_buffer(dict(graph=self.graph))
        self.graph.remove_nodes_from(("C", "D"))  # cleanup

        self.graph.add_edge("H", "A")  # create loop
        with pytest.raises(ValueError):
            write_read_buffer(dict(graph=self.graph))
