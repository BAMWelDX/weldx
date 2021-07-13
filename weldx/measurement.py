"""Contains measurement related classes and functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple  # noqa: F401
from warnings import warn

from networkx import draw, draw_networkx_edge_labels

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as ureg
from weldx.core import TimeSeries

if TYPE_CHECKING:  # pragma: no cover
    from pint import Quantity

    from weldx.core import MathematicalExpression


# measurement --------------------------------------------------------------------------


@dataclass
class Error:
    """Simple dataclass implementation for signal transformation errors."""

    deviation: Quantity = None


@dataclass
class Signal:
    """Simple dataclass implementation for measurement signals."""

    signal_type: str
    unit: str
    data: TimeSeries = None

    def __post_init__(self):
        """Perform some checks after construction."""
        if self.signal_type not in ["analog", "digital"]:
            raise ValueError(f"{self.signal_type} is an invalid signal type.")


@dataclass
class SignalTransformation:
    """Describes the transformation of a signal."""

    name: str
    error: Error
    func: MathematicalExpression = None
    type_transformation: str = None
    input_shape: Tuple = None
    output_shape: Tuple = None
    meta: str = None

    def __post_init__(self):
        """Perform some tests after construction."""
        if self.type_transformation is not None:
            self.type_transformation = self.type_transformation.upper()
            if self.type_transformation not in ["AA", "AD", "DA", "DD"]:
                raise ValueError(
                    f"Invalid type transformation: {self.type_transformation}\n"
                    "Valid values are 'AA', 'AD', 'DA' and 'DD'."
                )

        if self.func is None and self.type_transformation is None:
            raise ValueError("No transformation specified")

        if self.func is not None:
            self._evaluate_function()

    def _evaluate_function(self):
        """Evaluate the internal function."""
        variables = self.func.get_variable_names()
        if len(variables) != 1:
            raise ValueError("The provided function must have exactly one parameter")


@dataclass
class SignalSource:
    """Simple dataclass implementation for signal sources."""

    name: str
    output_signal: Signal
    error: Error


# equipment ----------------------------------------------------------------------------
@dataclass
class MeasurementEquipment:
    """Simple dataclass implementation for generic equipment."""

    name: str
    sources: List = field(default_factory=lambda: [])
    transformations: List = field(default_factory=lambda: [])

    def __post_init__(self):
        """Perform some data consistency checks."""
        sources = self.source_names
        if len(sources) != len(set(sources)):
            raise ValueError(
                "Two or more of the provided sources have identical names."
            )

        transformations = self.transformation_names
        if len(transformations) != len(set(transformations)):
            raise ValueError(
                "Two or more of the provided transformations have identical names"
            )

    def get_source(self, name: str) -> SignalSource:
        """Get a source by its name.

        Parameters
        ----------
        name :
            Name of the source

        Returns
        -------
        SignalSource :
            The requested source

        """
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"No source with name '{name}' found.")

    @property
    def source_names(self) -> List[str]:
        """Get the names of all sources.

        Returns
        -------
        List[str] :
            Names of all sources

        """
        return [source.name for source in self.sources]

    def get_transformation(self, name: str) -> SignalTransformation:
        """Get a transformation by its name.

        Parameters
        ----------
        name :
            Name of the transformation

        Returns
        -------
        SignalTransformation :
            The requested transformation

        """
        for transformation in self.transformations:
            if transformation.name == name:
                return transformation
        raise KeyError(f"No transformation with name '{name}' found.")

    @property
    def transformation_names(self) -> List[str]:
        """Get the names of all transformations.

        Returns
        -------
        List[str] :
            Names of all transformations

        """
        return [transformation.name for transformation in self.transformations]


# MeasurementChain ---------------------------------------------------------------------


class MeasurementChain:
    """Class that represents a measurement chain."""

    def __init__(
        self,
        name: str,
        source: SignalSource,
        signal_data: "TimeSeries" = None,
    ):
        """Create a new measurement chain.

        Parameters
        ----------
        name :
            Name of the measurement chain
        source :
            The source of the measurement chain
        signal_data :
            Measured data of the sources' signal

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.measurement import Error, MeasurementChain, Signal, SignalSource

        Create a signal source

        >>> current_source = SignalSource(name="Current sensor",
        ...                               error=Error(Q_(0.1, "percent")),
        ...                               output_signal=Signal(signal_type="analog",
        ...                                                    unit="V")
        ...                               )

        Create a measurement chain using the source

        >>> mc = MeasurementChain(name="Current measurement chain",
        ...                       source=current_source
        ...      )

        """
        from networkx import DiGraph

        self._name = name
        self._source = source
        self._source_equipment = None
        self._prev_added_signal = None
        self._graph = DiGraph()

        self._add_signal(node_id=source.name, signal=source.output_signal)
        if signal_data is not None:
            self.add_signal_data(signal_data)

    def __eq__(self, other: "MeasurementChain") -> bool:
        """Return `True` if two measurement chains are equal and `False` otherwise."""
        return (
            self._name == other._name
            and self._source == other._source
            and self._source_equipment == other._source_equipment
            and self._graph.nodes == other._graph.nodes
            and self._graph.edges == other._graph.edges
        )

    @classmethod
    def from_dict(cls, dictionary: Dict) -> "MeasurementChain":
        """Create a measurement chain from a dictionary.

        Parameters
        ----------
        dictionary :
            A dictionary containing all relevant data

        """
        mc = MeasurementChain(name=dictionary["name"], source=dictionary["data_source"])
        mc._graph = dictionary["graph"]
        mc._source_equipment = dictionary.get("source_equipment")
        for node in mc._graph.nodes:
            if len(list(mc._graph.successors(node))) == 0:
                mc._prev_added_signal = node
                break

        return mc

    @classmethod
    def from_equipment(
        cls, name, equipment: MeasurementEquipment, source_name=None
    ) -> "MeasurementChain":
        """Create a measurement chain from a piece of equipment that contains a source.

        Parameters
        ----------
        name :
            Name of the measurement chain
        equipment :
            A piece of equipment that contains one or more sources
        source_name :
            In case the provided equipment has more than one source, the desired one can
            be selected by name. If it only has a single source, this parameter can be
            set to `None` (default)

        Returns
        -------
        MeasurementChain :
            New measurement chain

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.measurement import Error, MeasurementChain,\
        MeasurementEquipment, Signal, SignalSource

        Create a signal source

        >>> current_source = SignalSource(name="Current sensor",
        ...                               error=Error(Q_(0.1, "percent")),
        ...                               output_signal=Signal(signal_type="analog",
        ...                                                    unit="V")
        ...                               )

        Create the equipment

        >>> current_sensor = MeasurementEquipment(name="Current Sensor",
        ...                                       sources=[current_source]
        ...                                       )

        Create a measurement chain using the equipment

        >>> mc = MeasurementChain.from_equipment(name="Current measurement chain",
        ...                                      equipment=current_sensor
        ...                                      )

        """
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

        mc = cls(name, source)
        mc._source_equipment = equipment
        return mc

    @classmethod
    def from_parameters(
        cls,
        name: str,
        source_name: str,
        source_error: Error,
        output_signal_type: str,
        output_signal_unit: str,
        signal_data: "TimeSeries" = None,
    ) -> "MeasurementChain":
        """Create a new measurement chain without providing a `SignalSource` instance.

        Parameters
        ----------
        name :
            Name of the measurement chain
        source_name :
            Name of the source
        source_error :
            Error of the source
        output_signal_type :
            Type of the output signal ('analog' or 'digital')
        output_signal_unit :
            Unit of the output signal
        signal_data :
            Measured data of the sources' signal

        Returns
        -------
        MeasurementChain :
            New measurement chain

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.measurement import Error, MeasurementChain

        >>> mc = MeasurementChain.from_parameters(
        ...          name="Current measurement chain",
        ...          source_error=Error(deviation=Q_(0.5, "percent")),
        ...          source_name="Current sensor",
        ...          output_signal_type="analog",
        ...          output_signal_unit="V"
        ...      )

        """
        source = SignalSource(
            source_name,
            Signal(output_signal_type, output_signal_unit, signal_data),
            source_error,
        )
        return cls(name, source)

    def _add_signal(
        self,
        node_id: str,
        signal: Signal = None,
    ):
        """Add a new signal node to the internal graph.

        Parameters
        ----------
        node_id :
            The name that will be used for the internal graphs' node. This is identical
            to the name of the signals source.
        signal :
            The signal that should be added

        """
        self._raise_if_node_exist(node_id)

        self._graph.add_node(node_id, signal=signal)
        self._prev_added_signal = node_id

    def _check_and_get_node_name(self, node_name: str) -> str:
        """Check if a node is part of the internal graph and return its name.

        If no name is provided, the name of the last added node is returned.

        """
        if node_name is None:
            return self._prev_added_signal
        self._raise_if_node_does_not_exist(node_name)
        return node_name

    @classmethod
    def _determine_output_signal(
        cls,
        transformation: SignalTransformation,
        input_signal: Signal,
        data: "TimeSeries",
    ) -> Signal:
        """Create a signal that is produced by the provided transformation.

        Parameters
        ----------
        transformation :
            A transformation
        input_signal :
            The input signal to the transformation
        data :
            A set of data that is measured after the transformation was applied. It is
            added to the returned signal

        Returns
        -------
        Signal :
            The resulting signal

        """
        return Signal(
            signal_type=cls._determine_output_signal_type(
                transformation.type_transformation, input_signal.signal_type
            ),
            unit=cls._determine_output_signal_unit(
                transformation.func, input_signal.unit
            ),
            data=data,
        )

    @staticmethod
    def _determine_output_signal_type(type_transformation: str, input_type: str) -> str:
        """Determine the type of a transformations' output signal.

        Parameters
        ----------
        type_transformation :
            The string describing the type transformation
        input_type :
            The type of the input signal

        Returns
        -------
        str :
            The type of the output signal

        """
        if type_transformation is not None:
            lookup = dict(A="analog", D="digital")
            exp_input_type = lookup.get(type_transformation[0])

            if input_type != exp_input_type:
                raise ValueError(
                    f"The transformation expects an {exp_input_type} as input signal, "
                    f"but the current signal is {input_type}."
                )
            return lookup.get(type_transformation[1])

        return input_type

    @staticmethod
    def _determine_output_signal_unit(
        func: "MathematicalExpression", input_unit: str
    ) -> str:
        """Determine the unit of a transformations' output signal.

        Parameters
        ----------
        func :
            The function describing the transformation
        input_unit :
            The unit of the input signal

        Returns
        -------
        str:
            Unit of the transformations' output signal

        """
        if func is not None:
            variables = func.get_variable_names()
            if len(variables) != 1:
                raise ValueError("The provided function must have exactly 1 parameter")

            try:
                test_output = func.evaluate(**{variables[0]: Q_(1, input_unit)})
            except Exception as e:
                raise ValueError(
                    "The provided function is incompatible with the input signals unit."
                    f" \nThe test raised the following exception:\n{e}"
                )
            return str(test_output.units)

        return input_unit

    def _raise_if_data_exist(self, signal_source_name: str):
        """Raise an error if the signal from the passed source already has data."""
        self._raise_if_node_does_not_exist(signal_source_name)
        if self._graph.nodes[signal_source_name]["signal"].data is not None:
            raise KeyError(
                "The measurement chain already contains data for "
                f"'{signal_source_name}'."
            )

    def _raise_if_node_exist(self, node_id: str):
        """Raise en error if the graph already contains a node with the passed id."""
        if node_id in self._graph.nodes:
            raise KeyError(
                f"The internal graph already contains a node with the id {node_id}"
            )

    def _raise_if_node_does_not_exist(self, node: str):
        """Raise a `KeyError` if the specified node does not exist."""
        if node not in self._graph.nodes:
            raise KeyError(f"No signal with source '{node}' found")

    def add_signal_data(self, data: TimeSeries, signal_source: str = None):
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
        data: "TimeSeries" = None,
        input_signal_source: str = None,
    ):
        """Add a transformation from an `SignalTransformation` instance.

        Parameters
        ----------
        transformation :
            The class containing the transformation data.
        data :
            A set of measurement data that is associated with the output signal of the
            transformation
        input_signal_source :
            The source of the signal that should be used as input of the transformation.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.core import MathematicalExpression
        >>> from weldx.measurement import Error, MeasurementChain, SignalTransformation

        >>> mc = MeasurementChain.from_parameters(
        ...          name="Current measurement chain",
        ...          source_error=Error(deviation=Q_(0.5, "percent")),
        ...          source_name="Current sensor",
        ...          output_signal_type="analog",
        ...          output_signal_unit="V"
        ...      )

        Create a mathematical expression that accepts a quantity with volts as unit and
        that returns a dimentsionless quantity.

        >>> func = MathematicalExpression(expression="a*x + b",
        ...                               parameters=dict(a=Q_(5, "1/V"), b=Q_(1, ""))
        ...                               )

        Use the mathematical expression to create a signal transformation which also
        performs a analog-digital conversion.

        >>> current_ad_transform = SignalTransformation(
        ...                            name="Current AD conversion",
        ...                            error=Error(deviation=Q_(1,"percent")),
        ...                            func=func,
        ...                            type_transformation="AD"
        ...                            )

        Add the transformation to the measurement chain.

        >>> mc.add_transformation(current_ad_transform)


        """
        input_signal_source = self._check_and_get_node_name(input_signal_source)
        input_signal = self._graph.nodes[input_signal_source]["signal"]
        output_signal = self._determine_output_signal(
            transformation, input_signal, data
        )

        self._add_signal(
            node_id=transformation.name,
            signal=output_signal,
        )
        self._graph.add_edge(
            input_signal_source, transformation.name, transformation=transformation
        )

    def add_transformation_from_equipment(
        self,
        equipment: MeasurementEquipment,
        data=None,
        input_signal_source: str = None,
        transformation_name=None,
    ):
        """Add a transformation from a piece of equipment.

        Parameters
        ----------
        equipment :
            The equipment that contains the transformation
        data :
            A set of measurement data that is associated with the output signal of the
            transformation
        input_signal_source :
            The source of the signal that should be used as input of the transformation.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.
        transformation_name :
            In case the provided piece of equipment contains multiple transformations,
            this parameter can be used to select one by name. If it only contains a
            single transformation, this parameter can be set to ´None´ (default)

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.core import MathematicalExpression
        >>> from weldx.measurement import Error, MeasurementChain,\
        MeasurementEquipment, SignalTransformation

        >>> mc = MeasurementChain.from_parameters(
        ...          name="Current measurement chain",
        ...          source_error=Error(deviation=Q_(0.5, "percent")),
        ...          source_name="Current sensor",
        ...          output_signal_type="analog",
        ...          output_signal_unit="V"
        ...      )

        Create a mathematical expression that accepts a quantity with volts as unit and
        that returns a dimentsionless quantity.

        >>> func = MathematicalExpression(expression="a*x + b",
        ...                               parameters=dict(a=Q_(5, "1/V"), b=Q_(1, ""))
        ...                               )

        Use the mathematical expression to create a signal transformation which also
        performs a analog-digital conversion.

        >>> current_ad_transform = SignalTransformation(
        ...                            name="Current AD conversion",
        ...                            error=Error(deviation=Q_(1,"percent")),
        ...                            func=func,
        ...                            type_transformation="AD"
        ...                            )

        Create a new equipment that performs the transformation

        >>> current_ad_converter = MeasurementEquipment(
        ...                            name="Current AD converter",
        ...                            transformations=[current_ad_transform]
        ...                        )

        Use the equipment to add the transformation to the measurement chain.

        >>> mc.add_transformation_from_equipment(current_ad_converter)

        """
        if len(equipment.transformations) > 1:
            if transformation_name is None:
                raise ValueError(
                    "The equipment has multiple transformations. Specify the "
                    "desired one by providing a 'transformation_name'."
                )
            transformation = equipment.get_transformation(transformation_name)
        elif len(equipment.transformations) == 1:
            transformation = equipment.transformations[0]
        else:
            raise ValueError("The equipment does not have a transformation.")

        input_signal_source = self._check_and_get_node_name(input_signal_source)
        self.add_transformation(transformation, data, input_signal_source)

        edge = (input_signal_source, self._prev_added_signal)
        self._graph.edges[edge]["equipment"] = equipment

    def create_transformation(
        self,
        name: str,
        error: Error,
        output_signal_type: str = None,
        output_signal_unit: str = None,
        func: "MathematicalExpression" = None,
        data: "TimeSeries" = None,
        input_signal_source: str = None,
    ):
        """Create and add a transformation to the measurement chain.

        Parameters
        ----------
        name :
            Name of the transformation
        error :
            The error of the transformation
        output_signal_type :
            Type of the output signal (analog or digital)
        output_signal_unit :
            Unit of the output signal. If a function is provided, it is not necessary to
            provide this parameter since it can be derived from the function. In case
            both, the function and the unit are provided, an exception is raised if a
            mismatch is dimensionality is detected. This functionality may be used as
            extra safety layer. If no function is provided, a simple unit conversion
            function is created.
        func :
            A function describing the transformation. The provided value interacts
            with the 'output_signal_unit' parameter as described in its documentation
        data :
            A set of measurement data that is associated with the output signal of the
            transformation
        input_signal_source :
            The source of the signal that should be used as input of the transformation.
            If `None` is provided, the name of the last added transformation (or the
            source, if no transformation was added to the chain) is used.

        Examples
        --------
        >>> from weldx import Q_
        >>> from weldx.core import MathematicalExpression
        >>> from weldx.measurement import Error, MeasurementChain, SignalTransformation

        >>> mc = MeasurementChain.from_parameters(
        ...          name="Current measurement chain",
        ...          source_error=Error(deviation=Q_(0.5, "percent")),
        ...          source_name="Current sensor",
        ...          output_signal_type="analog",
        ...          output_signal_unit="V"
        ...      )

        Create a mathematical expression that accepts a quantity with volts as unit and
        that returns a dimentsionless quantity.

        >>> func = MathematicalExpression(expression="a*x + b",
        ...                               parameters=dict(a=Q_(5, "1/V"), b=Q_(1, ""))
        ...                               )

        Use the mathematical expression to create a new transformation which also
        performs a analog-digital conversion.

        >>> mc.create_transformation(name="Current AD conversion",
        ...                          error=Error(deviation=Q_(1,"percent")),
        ...                          func=func,
        ...                          output_signal_type="digital"
        ...                          )

        """
        if output_signal_type is None and output_signal_unit is None and func is None:
            warn("The created transformation does not perform any transformations.")

        input_signal_source = self._check_and_get_node_name(input_signal_source)
        input_signal = self._graph.nodes[input_signal_source]["signal"]
        if output_signal_type is None:
            output_signal_type = input_signal.signal_type
        type_tf = f"{input_signal.signal_type[0]}{output_signal_type[0]}".upper()
        if output_signal_unit is not None:
            if func is not None:
                if not ureg.is_compatible_with(
                    output_signal_unit,
                    self._determine_output_signal_unit(func, input_signal.unit),
                ):
                    raise ValueError(
                        "The unit of the provided functions output has not the same "
                        f"dimensionality as {output_signal_unit}"
                    )
            else:
                from weldx.core import MathematicalExpression

                if output_signal_unit == "":
                    output_signal_unit = 1
                unit_conversion = f"{output_signal_unit}/{str(input_signal.unit)}"
                func = MathematicalExpression(
                    "a*x",
                    parameters={"a": Q_(1, unit_conversion)},
                )

        transformation = SignalTransformation(name, error, func, type_tf)
        self.add_transformation(transformation, data, input_signal_source)

    def get_equipment(self, signal_source: str) -> MeasurementEquipment:
        """Get the equipment that produced a signal.

        Parameters
        ----------
        signal_source :
            Source of the signal.

        Returns
        -------
        MeasurementEquipment :
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

    def get_signal_data(self, source_name: str = None) -> "TimeSeries":
        """Get the data from a signal.

        Parameters
        ----------
        source_name :
            Name of the data's source, e.g. a transformation or the source of the
            measurement chain. If `None` is provided, the data of the last added
            transformation is returned, if there is one.

        Returns
        -------
        TimeSeries :
            The requested data

        """
        source_name = self._check_and_get_node_name(source_name)
        signal = self._graph.nodes[source_name]["signal"]
        if signal.data is None:
            raise KeyError(f"There is no data for the source: '{source_name}'")
        return signal.data

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

    def to_dict(self) -> Dict:
        """Get the content of the measurement chain as dictionary.

        Returns
        -------
        Dict:
            Content of the measurement chain as dictionary.

        """
        return dict(
            name=self._name,
            data_source=self._source,
            source_equipment=self._source_equipment,
            graph=self._graph,
        )

    @property
    def signals(self):
        """Get a list of all signals."""
        return [self._graph.nodes[node]["signal"] for node in self._graph.nodes]

    @property
    def source(self) -> SignalSource:
        """Return the source of the measurement chain."""
        return self._source

    @property
    def source_name(self) -> str:
        """Get the name of the source.

        Returns
        -------
        str :
            Name of the source

        """
        return self._source.name

    @property
    def transformations(self):
        """Get a list of all transformations."""
        return [self._graph.edges[edge]["transformation"] for edge in self._graph.edges]

    @property
    def transformation_names(self) -> List:
        """Get the names of all transformations.

        Returns
        -------
        List :
            A list containing all transformation names

        """
        return [
            self._graph.edges[edge]["transformation"].name for edge in self._graph.edges
        ]

    @property
    def output_signal(self) -> Signal:
        """Get the output signal of the measurement chain."""
        return self.signals[-1]


@dataclass
class Measurement:
    """Simple dataclass implementation for generic measurements."""

    name: str
    data: List[TimeSeries]
    measurement_chain: MeasurementChain
