import networkx as nx

import weldx
from weldx.asdf.tags.weldx.core.graph import (
    MeasurementChain,
    SignalClass,
    SignalTransform,
)

G = nx.DiGraph()
G.add_edges_from(
    [("A", "B"), ("A", "C"), ("A", "F"), ("D", "C"), ("B", "H"), ("X", "A")]
)
nx.draw(G, with_labels=True)

signal = SignalClass("The Signal", 12.3)
nx.set_node_attributes(G, signal, "signal")

signal_transform = SignalTransform(func="a*b + c")
nx.set_edge_attributes(G, signal_transform, "signal_transform")

m = MeasurementChain(graph=G)

buff = weldx.asdf.util.write_buffer(dict(measurement_chain=m))
tree = weldx.asdf.util.read_buffer(buff)


G2 = tree["measurement_chain"].graph

assert sorted(G2.edges) == sorted(G.edges)
assert sorted(G2.nodes) == sorted(G.nodes)

for node in G:
    assert G.nodes[node] == G2.nodes[node]

for edge in G.edges:
    assert G.edges[edge] == G2.edges[edge]
