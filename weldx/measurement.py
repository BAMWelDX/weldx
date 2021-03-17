"""Contains measurement related classes and functions."""

from dataclasses import dataclass, field
from typing import List, Union  # noqa: F401

import matplotlib.pyplot as plt
import xarray as xr
from networkx import DiGraph, draw, draw_networkx_edge_labels


# measurement --------------------------------------------------------------------------
@dataclass
class Data:
    """Simple dataclass implementation for measurement data."""

    name: str
    data: xr.DataArray


@dataclass
class Error:
    """Simple dataclass implementation for signal transformation errors."""

    deviation: float


@dataclass
class Signal:
    """Simple dataclass implementation for measurement signals."""

    signal_type: str
    unit: str
    data: Union[Data, None]


@dataclass
class DataTransformation:
    """Simple dataclass implementation for signal transformations."""

    name: str
    input_signal: Signal
    output_signal: Signal
    error: Error
    func: str = None
    meta: str = None


@dataclass
class Source:
    """Simple dataclass implementation for signal sources."""

    name: str
    output_signal: Signal
    error: Error


@dataclass
class MeasurementChain:
    """Simple dataclass implementation for measurement chains."""

    name: str
    data_source: Source
    data_processors: List = field(default_factory=lambda: [])

    def plot(self, axes=None):
        def _signal_label(signal):
            return f"{signal.signal_type}\n[{signal.unit}]"

        if axes is None:
            _, axes = plt.subplots(nrows=1, figsize=(12, 6))

        axes.set_ylim(0, 1)
        axes.set_title(self.name, fontsize=20, fontweight="bold")

        graph = DiGraph()

        num_nodes = len(self.data_processors) + 1
        delta_pos = 2 / num_nodes
        current_pos = delta_pos / 2

        current_node = "node_0"
        previous_node = current_node
        graph.add_node(previous_node)

        node_list = [current_node]
        node_color_list = ["#55ff55"]
        labels = {current_node: _signal_label(self.data_source.output_signal)}
        positions = {current_node: (current_pos, 0.75)}

        edge_labels = {}

        for i, processor in enumerate(self.data_processors):
            current_pos += delta_pos
            current_node = f"node_{i+1}"
            node_list.append(current_node)
            node_color_list.append("#55ff55")
            graph.add_node(current_node)
            labels[current_node] = _signal_label(processor.output_signal)
            positions[current_node] = (current_pos, 0.75)

            edge = (previous_node, current_node)
            graph.add_edge(*edge)
            edge_labels[edge] = f"{processor.name}"

            if processor.output_signal.data is not None:
                data_name = processor.output_signal.data.name
                graph.add_node(data_name)
                node_list.append(data_name)
                node_color_list.append("#ffff55")
                labels[data_name] = data_name
                positions[data_name] = (current_pos, 0.25)
                graph.add_edge(current_node, data_name)

            previous_node = current_node

        draw(
            graph,
            positions,
            axes,
            nodelist=node_list,
            with_labels=True,
            font_weight="bold",
            font_color="k",
            labels=labels,
            node_size=3000,
            node_color=node_color_list,
        )
        draw_networkx_edge_labels(graph, positions, edge_labels, ax=axes)


@dataclass
class Measurement:
    """Simple dataclass implementation for generic measurements."""

    name: str
    data: Data
    measurement_chain: MeasurementChain


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """Simple dataclass implementation for generic equipment."""

    name: str
    sources: List = field(default_factory=lambda: [])
    data_transformations: List = field(default_factory=lambda: [])
