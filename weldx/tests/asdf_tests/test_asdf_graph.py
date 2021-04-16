"""Test graph serializations."""
import unittest
from uuid import uuid4

import networkx as nx
import pytest

from weldx.asdf.util import write_read_buffer

# --------------------------------------------------------------------------------------
# DiGraph
# --------------------------------------------------------------------------------------


class TestDiGraph(unittest.TestCase):
    def setUp(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from(
            [("A", "B"), ("A", "C"), ("A", "F"), ("D", "C"), ("B", "H"), ("X", "A")]
        )
        nx.set_node_attributes(g, 3.14, "node_attr")
        nx.set_edge_attributes(g, 42, "edge_attr")
        self.graph = g

        id_a, id_c = uuid4(), uuid4()
        g2 = nx.relabel.relabel_nodes(g, {"A": id_a, "C": id_c}, copy=True)
        setattr(g2, "_wx_keep_uuid_name", True)
        self.graph_uuid = g2

    @staticmethod
    def _assert_roundtrip(g):
        data = write_read_buffer(dict(graph=g))
        g2 = data["graph"]

        for node in g:
            assert g.nodes[node] == g2.nodes[node]

        for edge in g.edges:
            assert g.edges[edge] == g2.edges[edge]

        assert set(g.nodes) == set(g2.nodes)
        assert set(g.edges) == set(g2.edges)

    def test_graph_roundtrip(self):
        for g in [self.graph, self.graph_uuid]:
            self._assert_roundtrip(g)

    def test_graph_exceptions(self):
        self.graph.remove_edge("A", "C")  # two trees in graph
        with pytest.raises(ValueError):
            write_read_buffer(dict(graph=self.graph))
        self.graph.remove_nodes_from(("C", "D"))  # cleanup

        self.graph.add_edge("H", "A")  # create loop
        with pytest.raises(ValueError):
            write_read_buffer(dict(graph=self.graph))

        with pytest.raises(KeyError):
            setattr(self.graph_uuid, "_wx_keep_uuid_name", False)
            g = self.graph_uuid
            self._assert_roundtrip(g)
