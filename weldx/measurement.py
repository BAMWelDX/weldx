"""Contains measurement related classes and functions."""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple, Union  # noqa: F401

import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from networkx import DiGraph

    from weldx.core import MathematicalExpression, TimeSeries


# measurement --------------------------------------------------------------------------
@dataclass
class Data:
    """Simple dataclass implementation for measurement data."""

    name: str
    data: xr.DataArray  # skipcq: PTC-W0052

    def __eq__(self, other):
        """Check for equality with other object."""
        if not isinstance(other, Data):
            return False
        return self.name == other.name and self.data.identical(other.data)


@dataclass
class Error:
    """Simple dataclass implementation for signal transformation errors."""

    deviation: float


@dataclass
class Signal:
    """Simple dataclass implementation for measurement signals."""

    signal_type: str
    unit: str
    data: Union[Data, None] = None

    def __post_init__(self):
        if self.signal_type not in ["analog", "digital"]:
            raise ValueError(f"{self.signal_type} is an invalid signal type.")


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
class SignalTransformation:
    name: str
    error: Error
    func: "MathematicalExpression" = None
    input_shape: Tuple = None
    output_shape: Tuple = None
    io_types: str = None

    def __post_init__(self):
        if self.io_types is not None:
            self.io_types = self.io_types.upper()
            if self.io_types not in ["AA", "AD", "DA", "DD"]:
                raise ValueError(
                    f"Invalid type transformation: {self.io_types}\n"
                    "Valid values are 'AA', 'AD', 'DA' and 'DD'."
                )

        if self.func is None and self.io_types is None:
            raise ValueError("No transformation specified")

        if self.func is not None:
            self._evaluate_function()

    def _evaluate_function(self):
        from weldx import Q_

        variables = self.func.get_variable_names()
        if len(variables) != 1:
            raise ValueError("The provided function must have exactly one parameter")
        variable_name = variables[0]

        # test_input = Q_(1, self.input_signal.unit)
        # try:
        #    test_output = self.func.evaluate(**{variable_name: test_input})
        # except Exception as e:
        #    raise ValueError(
        #        "The provided function is incompatible with the input signals unit. \n"
        #        f"The test raised the following exception:\n{e}"
        #    )
        # if Q_(1, self.output_signal.unit).units != test_output.units:
        #    raise ValueError(
        #        "The test result of the provided function has a different unit than "
        #        "the specified output signal.\n"
        #        f"output_signal: {self.output_signal.unit}\n"
        #        f"test_result  : {test_output.units}\n"
        #    )


@dataclass
class SignalSource:
    """Simple dataclass implementation for signal sources."""

    name: str
    output_signal: Signal
    error: Error


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """Simple dataclass implementation for generic equipment."""

    name: str
    sources: List = field(default_factory=lambda: [])
    data_transformations: List = field(default_factory=lambda: [])

    def get_source(self, name):
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"No source with name '{name}' found.")

    @property
    def source_names(self):
        return [source.name for source in self.sources]

    def get_transformation(self, name):
        for transformation in self.data_transformations:
            if transformation.name == name:
                return transformation
        raise KeyError(f"No transformation with name '{name}' found.")

    @property
    def transformation_names(self):
        return [transformation.name for transformation in self.data_transformations]


# DRAFT SECTION START ##################################################################

# todo: - which classes can be removed?
#       - tutorial

#       - transformation -> only units
#       - source -> signal


class MeasurementChain:
    """Simple dataclass implementation for measurement chains."""

    def __init__(
        self,
        name: str,
        source: SignalSource,
        signal_data: xr.DataArray = None,
    ):
        """Create a new measurement chain.

        Parameters
        ----------
        name :
            Name of the measurement chain
        source :
            The source of the measurement chain

        """
        from networkx import DiGraph

        self._name = name
        self._source = source
        self._source_equipment = None
        self._prev_added_signal = None

        self._graph = DiGraph()
        self._add_signal(
            node_id=source.name,
            signal=source.output_signal,
        )
        if signal_data is not None:
            self.add_signal_data(signal_data)  # todo : test

    @classmethod
    def from_parameters(
        cls,
        name: str,
        source_name: str,
        source_error: Error,
        output_signal_type: str = None,
        output_signal_unit: str = None,
        output_signal: Signal = None,
        signal_data: xr.DataArray = None,
    ):
        source = SignalSource(
            source_name,
            Signal(output_signal_type, output_signal_unit, signal_data),
            source_error,
        )
        return MeasurementChain(name, source)

    @staticmethod
    def construct_from_source(name, source: SignalSource):

        return MeasurementChain(
            name=name,
            source_name=source.name,
            source_error=source.error,
            output_signal_type=None,
            output_signal_unit=None,
            output_signal=source.output_signal,
            signal_data=None,
        )

    @staticmethod
    def construct_from_equipment(name, equipment: GenericEquipment, source_name=None):
        if len(equipment.sources) > 1:
            if source_name is None:
                raise ValueError(
                    "The equipment has multiple sources. Specify the "
                    "desired one by providing a 'source_name'."
                )
            source = equipment.get_source(source_name)
        elif len(equipment.sources) == 1:
            source = equipment.sources[0]
        else:
            raise ValueError("The equipment does not have a source.")

        mc = MeasurementChain(name, source)
        mc._source_equipment = equipment
        return mc

    @staticmethod
    def construct_from_tree(tree: Dict) -> "MeasurementChain":
        mc = MeasurementChain(
            name=tree["name"],
            source_name="source",
            source_error=Error(1),
            output_signal_type="analog",
            output_signal_unit="V",
        )
        # todo: implement correct version, when schema is ready
        return mc

    def _add_signal(
        self,
        node_id: str,
        signal_type: str = None,
        unit: str = None,
        signal: Signal = None,
    ):
        """Add a new signal node to internal graph.

        Parameters
        ----------
        node_id :
            The name that will be used for the internal graphs' node. This is identical
            to the name of the signals source.
        signal_type :
            Type of the signal (analog or digital)
        unit :
            Unit of the signal

        """
        self._raise_if_node_exist(node_id)
        signal = self._create_signal(signal_type, unit, signal)

        self._graph.add_node(node_id, signal=signal)
        self._prev_added_signal = node_id

    def _raise_if_node_exist(self, node_id: str):
        """Raise en error if the graph already contains a node with the passed id.

        Parameters
        ----------
        node_id :
            Name that should be checked

        """
        if node_id in self._graph.nodes:
            raise KeyError(
                f"The internal graph already contains a node with the id {node_id}"
            )

    def _raise_if_node_does_not_exist(self, node: str):
        """Raise a `KeyError` if the specified node does not exist.

        Parameters
        ----------
        node :
            Name of the node that should be searched for

        """
        if node not in self._graph.nodes:
            raise KeyError(f"No signal with source '{node}' found")

    def _raise_if_data_exist(self, source_name: str):
        """Raise an error if a data set with the passed name already exists.

        Parameters
        ----------
        source_name :
            Name of the data's source, e.g. a transformation or the source of the
            measurement chain

        """
        self._raise_if_node_does_not_exist(source_name)
        if "data" in self._graph.nodes[source_name]:
            raise KeyError("The measurement chain already contains data for '{name}'.")

    def _check_and_get_node_name(self, node_name: str) -> str:
        """Check if a node is part of the internal graph and return its name.

        If no name is provided, the name of the last added node is returned.

        Parameters
        ----------
        node_name :
            Name of the node that should be checked.

        """
        if node_name is None:
            return self._prev_added_signal
        else:
            self._raise_if_node_does_not_exist(node_name)
        return node_name

    @staticmethod
    def _create_signal(signal_type: str, unit: str, signal: Signal) -> Signal:
        if signal is None:
            if signal_type is None or unit is None:
                raise ValueError(
                    "You need to provide a signal type AND unit. Alternatively, an "
                    "already existing Signal can be passed."
                )
            return Signal(signal_type, unit)
        else:
            if signal_type is not None and unit is not None:
                warnings.warn(
                    "Provided signal type and/or unit is ignored because an existing "
                    "signal was passed."
                )
            return signal

    def _signal_after_transformation(self, transformation, input_signal, data):
        from weldx import Q_

        if transformation.func is not None:
            variables = transformation.func.get_variable_names()
            if len(variables) != 1:
                raise ValueError(
                    "The provided function must have exactly one parameter"
                )
            variable_name = variables[0]

            test_input = Q_(1, input_signal.unit)
            try:
                test_output = transformation.func.evaluate(
                    **{variable_name: test_input}
                )
            except Exception as e:
                raise ValueError(
                    "The provided function is incompatible with the input signals unit."
                    f" \nThe test raised the following exception:\n{e}"
                )
            output_unit = str(
                test_output.units
            )  # just pass pint unit without string conversion?
        else:
            output_unit = input_signal.unit

        if transformation.io_types is not None:
            if transformation.io_types[1] == "A":
                output_type = "analog"
            else:
                output_type = "digital"
        else:
            output_type = input_signal.signal_type

        return Signal(signal_type=output_type, unit=output_unit, data=data)

    # todo: rename to create_transformation
    def create_transformation(
        self,
        name: str,
        error: Error,
        output_signal_type: str = None,
        func: "MathematicalExpression" = None,
        input_signal_source: str = None,
        data=None,
        # expected output unit as optional safety parameter?
    ):
        """Add transformation to the measurement chain.

        Parameters
        ----------
        name :
            Name of the transformation
        error :
            The error of the transformation
        output_signal_type :
            Type of the output signal (analog or digital)
        output_signal_unit :
            Unit of the output signal
        func :
            A function describing the transformation
        input_signal_source :
            The source of the signal that should be used as input of the transformation.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.

        """
        input_signal_source = self._check_and_get_node_name(input_signal_source)
        input_signal = self._graph.nodes[input_signal_source]["signal"]

        type_tf = f"{input_signal.signal_type[0]}{output_signal_type[0]}".upper()

        transformation = SignalTransformation(name, error, func, io_types=type_tf)

        self.add_transformation(transformation, data, input_signal_source)

    def add_transformation_from_equipment(
        self,
        equipment: GenericEquipment,
        data=None,
        input_signal_source: str = None,
        transformation_name=None,
    ):
        if len(equipment.data_transformations) > 1:
            if transformation_name is None:
                raise ValueError(
                    "The equipment has multiple transformations. Specify the "
                    "desired one by providing a 'transformation_name'."
                )
            transformation = equipment.get_transformation(transformation_name)
        elif len(equipment.data_transformations) == 1:
            transformation = equipment.data_transformations[0]
        else:
            raise ValueError("The equipment does not have a transformation.")

        input_signal_source = self._check_and_get_node_name(input_signal_source)
        self.add_transformation(transformation, data, input_signal_source)
        self._graph.edges[(input_signal_source, self._prev_added_signal)][
            "equipment"
        ] = equipment

    def add_signal_data(self, data: "TimeSeries", signal_source: str = None):
        """Add data to a signal.

        Parameters
        ----------
        data :
            The data that should be added
        signal_source :
            The source of the signal that the data should be attached to.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.

        """
        signal_source = self._check_and_get_node_name(signal_source)
        self._raise_if_data_exist(signal_source)
        signal = self._graph.nodes[signal_source]["signal"]
        signal.data = data

    def add_transformation(
        self,
        transformation: SignalTransformation,
        data=None,
        input_signal_source: str = None,
    ):
        """Add a transformation from an `SignalTransformation` instance.

        Parameters
        ----------
        transformation :
            The class containing the transformation data.
        input_signal_source :
            The source of the signal that should be used as input of the transformation.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.

        """
        input_signal_source = self._check_and_get_node_name(input_signal_source)
        input_signal = self._graph.nodes[input_signal_source]["signal"]
        output_signal = self._signal_after_transformation(
            transformation, input_signal, data
        )

        self._add_signal(
            node_id=transformation.name,
            signal=output_signal,
        )
        self._graph.add_edge(
            input_signal_source, transformation.name, transformation=transformation
        )

    def get_signal_data(self, source_name: str = None) -> xr.DataArray:
        """

        Parameters
        ----------
        source_name :
            Name of the data's source, e.g. a transformation or the source of the
            measurement chain. If `None` is provided, the data of the last added
            transformation is returned, if there is one.

        Returns
        -------
        xarray.DataArray :
            The requested data

        """
        source_name = self._check_and_get_node_name(source_name)
        signal = self._graph.nodes[source_name]["signal"]
        if signal.data is None:
            raise KeyError(f"There is no data for the source: '{source_name}'")
        return signal.data

    @property
    def data_names(self) -> List[str]:
        """Get the names of all attached data sets.

        Returns
        -------
        List[str] :
            List of the names from all attached data sets

        """
        return [
            attr["data_name"]
            for _, attr in self._graph.nodes.items()
            if "data_name" in attr
        ]

    def get_equipment(self, signal_source: str) -> GenericEquipment:
        """Get the equipment that produced a signal.

        Parameters
        ----------
        signal_source :
            Source of the signal.

        Returns
        -------
        GenericEquipment :
            The requested equipment

        """
        if signal_source == self._source.name:
            return self._source_equipment
        for edge in self._graph.edges:
            if edge[1] == signal_source:
                return self._graph.edges[edge].get("equipment")
        raise KeyError(f"No transformation with name '{signal_source}' found")

    def get_signal(self, signal_source: str) -> Signal:
        """Get a signal.

        Parameters
        ----------
        signal_source :
            Name of the signals source.

        Returns
        -------
        Signal :
            The requested signal

        """
        self._raise_if_node_does_not_exist(signal_source)
        return self._graph.nodes[signal_source]["signal"]

    def get_transformation(self, name: str) -> SignalTransformation:
        """Get a transformation.

        Parameters
        ----------
        name :
            Name of the transformation

        Returns
        -------
        SignalTransformation :
            The requested transformation

        """
        for edge in self._graph.edges:
            if edge[1] == name:
                return self._graph.edges[edge]["transformation"]

        raise KeyError(f"No transformation with name '{name}' found")

    def plot(self, axes=None):
        """Plot the measurement chain.

        Parameters
        ----------
        axes :
            Matplotlib axes object that should be drawn to. If None is provided, this
            function will create one.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object the graph has been drawn to

        """
        import matplotlib.pyplot as plt
        from networkx import draw, draw_networkx_edge_labels

        def _transformation_label(name, transformation: SignalTransformation) -> str:
            text = name
            func = transformation.func
            error = transformation.error
            if func is not None:
                text += f"\n{func.expression}"
            if error is not None and error.deviation != 0.0:
                text += f"\nerr: {error.deviation}"
            return text

        if axes is None:
            _, axes = plt.subplots(nrows=1, figsize=(12, 6))

        axes.set_ylim(0, 1)
        axes.set_title(self._name, fontsize=20, fontweight="bold")

        graph = self._graph.copy()

        data_labels = {}
        signal_labels = {}
        positions = {}

        c_node = self._source.name
        delta_pos = 2 / (len(graph.nodes) + 1)
        x_pos = delta_pos / 2

        # walk over the graphs nodes and collect necessary data for plotting
        while True:
            signal = graph.nodes[c_node]["signal"]
            signal_labels[c_node] = f"{signal.signal_type}\n[{signal.unit}]"
            positions[c_node] = (x_pos, 0.75)

            if signal.data is not None:
                data_name = f"{c_node}\ndata"
                graph.add_edge(c_node, data_name)
                data_labels[data_name] = data_name
                positions[data_name] = (x_pos, 0.25)

            successors = list(self._graph.successors(c_node))
            if len(successors) == 0:
                break
            if len(successors) > 1:
                raise ValueError(
                    "Signals with multiple transformations are currently "
                    "not supported by the plot function."
                )
            c_node = successors[0]
            x_pos += delta_pos

        # draw signal nodes and all edges
        draw(
            graph,
            positions,
            ax=axes,
            nodelist=self._graph.nodes(),
            with_labels=True,
            labels=signal_labels,
            font_weight="bold",
            font_color="k",
            node_size=3000,
            node_shape="s",
            node_color="#bbbbbb",
        )

        # draw data nodes
        data_node_list = [
            node for node in graph.nodes if node not in self._graph.nodes()
        ]

        draw(
            graph,
            positions,
            axes,
            nodelist=data_node_list,
            with_labels=True,
            labels=data_labels,
            font_weight="bold",
            font_color="k",
            edgelist=[],
            node_size=3000,
            node_color="#bbbbbb",
        )

        # draw edge labels
        edge_labels = {
            edge: _transformation_label(
                edge[1], self._graph.edges[edge]["transformation"]
            )
            for edge in self._graph.edges
        }

        draw_networkx_edge_labels(graph, positions, edge_labels, ax=axes)

        return axes


# DRAFT SECTION END ####################################################################


@dataclass
class Measurement:
    """Simple dataclass implementation for generic measurements."""

    name: str
    data: Data
    measurement_chain: MeasurementChain
