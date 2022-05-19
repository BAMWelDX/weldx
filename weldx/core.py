"""Collection of common classes and functions."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import pint
import sympy
import xarray as xr
from bidict import bidict

import weldx.util as ut
from weldx.constants import Q_, U_, UNITS_KEY
from weldx.time import Time, TimeDependent, types_time_like
from weldx.util import check_matplotlib_available

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes
    from xarray.core.coordinates import DataArrayCoordinates

    from weldx.types import UnitLike

__all__ = ["GenericSeries", "MathematicalExpression", "TimeSeries", "SpatialSeries"]

_me_parameter_types = Union[pint.Quantity, str, Tuple[pint.Quantity, str], xr.DataArray]


class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(
        self,
        expression: Union[sympy.Expr, str],
        parameters: _me_parameter_types = None,
    ):
        """Construct a MathematicalExpression.

        Parameters
        ----------
        expression
            A sympy expression or a string that can be evaluated as one.
        parameters :
            A dictionary containing constant values for variables of the
            expression.

        """
        if not isinstance(expression, sympy.Expr):
            expression = sympy.sympify(expression)
        if not isinstance(expression, sympy.Expr):
            raise TypeError("'expression' can't be converted to a sympy expression")
        self._expression: sympy.Expr = expression

        self.function = sympy.lambdify(
            tuple(self._expression.free_symbols), self._expression, "numpy"
        )

        self._parameters: dict[str, Union[pint.Quantity, xr.DataArray]] = {}
        if parameters is not None:
            self.set_parameters(parameters)

    def __repr__(self):
        """Give __repr__ output."""
        # todo: Add covered dimensions to parameters - coordinates as well?
        representation = (
            f"<MathematicalExpression>\n"
            f"Expression:\n\t{self._expression.__repr__()}\n"
            f"Parameters:\n"
        )
        if len(self._parameters) > 0:
            for parameter, value in self._parameters.items():
                if isinstance(value, xr.DataArray):
                    value = value.data
                representation += f"\t{parameter} = {value}\n"
        else:
            representation += "\tNone"
        return representation

    def __eq__(self, other):
        """Return the result of a structural equality comparison with another object.

        If the other object is not a 'MathematicalExpression' this function always
        returns 'False'.

        Parameters
        ----------
        other:
            Other object.

        Returns
        -------
        bool:
            'True' if the compared object is also a 'MathematicalExpression' and equal
            to this instance, 'False' otherwise

        """
        return self.equals(other, check_parameters=True, check_structural_equality=True)

    def equals(
        self,
        other: Any,
        check_parameters: bool = True,
        check_structural_equality: bool = False,
    ):
        """Compare the instance with another object for equality and return the result.

        If the other object is not a MathematicalExpression this function always returns
        'False'. The function offers the choice to compare for structural or
        mathematical equality by setting the 'structural_expression_equality' parameter
        accordingly. Additionally, the comparison can be limited to the expression only,
        if 'check_parameters' is set to 'False'.

        Parameters
        ----------
        other:
            Arbitrary other object.
        check_parameters
            Set to 'True' if the parameters should be included during the comparison. If
            'False', only the expression is checked for equality.
        check_structural_equality:
            Set to 'True' if the expression should be checked for structural equality.
            Set to 'False' if mathematical equality is sufficient.

        Returns
        -------
        bool:
            'True' if both objects are equal, 'False' otherwise

        """
        if isinstance(other, MathematicalExpression):
            if check_structural_equality:
                equality = self.expression == other.expression
            else:
                from sympy import simplify

                equality = simplify(self.expression - other.expression) == 0

            if check_parameters:
                from weldx.util import compare_nested

                equality = equality and compare_nested(
                    self._parameters, other.parameters
                )
            return equality
        return False

    def set_parameter(self, name, value):
        """Define an expression parameter as constant value.

        Parameters
        ----------
        name
            Name of the parameter used in the expression.
        value
            Parameter value. This can be number, array or pint.Quantity

        """
        self.set_parameters({name: value})

    # todo: Use kwargs here???
    def set_parameters(self, params: _me_parameter_types):
        """Set the expressions parameters.

        Parameters
        ----------
        params:
            Dictionary that contains the values for the specified parameters.

        """
        if not isinstance(params, dict):
            raise ValueError(f'"parameters" must be dictionary, got {type(params)}')

        variable_names = [str(v) for v in self._expression.free_symbols]

        for k, v in params.items():
            if k not in variable_names:
                raise ValueError(f'The expression does not have a parameter "{k}"')
            if isinstance(v, tuple):
                v = xr.DataArray(v[0], dims=v[1])
            if not isinstance(v, xr.DataArray):
                v = Q_(v)
            self._parameters[k] = v

    @property
    def num_parameters(self):
        """Get the expressions number of parameters.

        Returns
        -------
        int:
            Number of parameters.

        """
        return len(self._parameters)

    @property
    def num_variables(self):
        """Get the expressions number of free variables.

        Returns
        -------
        int:
            Number of free variables.

        """
        return len(self.expression.free_symbols) - len(self._parameters)

    @property
    def expression(self):
        """Return the internal sympy expression.

        Returns
        -------
        sympy.core.expr.Expr:
            Internal sympy expression

        """
        return self._expression

    @property
    def parameters(self) -> dict[str, Union[pint.Quantity, xr.DataArray]]:
        """Return the internal parameters dictionary.

        Returns
        -------
        Dict
            Internal parameters dictionary

        """
        return self._parameters

    def get_variable_names(self) -> list[str]:
        """Get a list of all expression variables.

        Returns
        -------
        List:
            List of all expression variables

        """
        return [
            str(var)
            for var in self._expression.free_symbols
            if str(var) not in self._parameters
        ]

    def evaluate(self, **kwargs) -> Any:
        """Evaluate the expression for specific variable values.

        Parameters
        ----------
        kwargs
            additional keyword arguments (variable assignment) to pass.

        Returns
        -------
        Any:
            Result of the evaluated function

        """
        intersection = set(kwargs).intersection(self._parameters)
        if len(intersection) > 0:
            raise ValueError(
                f"The variables {intersection} are already defined as parameters."
            )

        variables = {
            k: v if isinstance(v, xr.DataArray) else xr.DataArray(Q_(v))
            for k, v in kwargs.items()
        }

        parameters = {
            k: v if isinstance(v, xr.DataArray) else xr.DataArray(v)
            for k, v in self._parameters.items()
        }
        return self.function(**variables, **parameters)


# TimeSeries ---------------------------------------------------------------------------


class TimeSeries(TimeDependent):
    """Describes the behaviour of a quantity in time."""

    _valid_interpolations = [
        "step",
        "linear",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
    ]

    def __init__(
        self,
        data: Union[pint.Quantity, MathematicalExpression],
        time: types_time_like = None,
        interpolation: str = None,
        reference_time: pd.Timestamp = None,
    ):
        """Construct a TimSeries.

        Parameters
        ----------
        data:
            Either a pint.Quantity or a weldx.MathematicalExpression. If a mathematical
            expression is chosen, it is only allowed to have a single free variable,
            which represents time.
        time:
            An instance of pandas.TimedeltaIndex if a quantity is passed and 'None'
            otherwise.
        interpolation:
            A string defining the desired interpolation method. This is only relevant if
            a quantity is passed as data. Currently supported interpolation methods are:
            'step', 'linear'.

        """
        self._data: Union[MathematicalExpression, xr.DataArray] = None
        self._time_var_name = None  # type: str
        self._shape = None
        self._units = None
        self._interp_counter = 0
        self._reference_time = reference_time

        if isinstance(data, (pint.Quantity, xr.DataArray)):
            self._initialize_discrete(data, time, interpolation)
        elif isinstance(data, MathematicalExpression):
            self._init_expression(data)
        else:
            raise TypeError(f'The data type "{type(data)}" is not supported.')

    def __eq__(self, other: Any) -> bool:
        """Return the result of a structural equality comparison with another object.

        If the other object is not a 'TimeSeries' this function always returns 'False'.

        Parameters
        ----------
        other:
            Other object.

        Returns
        -------
        bool:
           'True' if the compared object is also a 'TimeSeries' and equal to
            this instance, 'False' otherwise

        """
        if not isinstance(other, TimeSeries):
            return False
        if not isinstance(self.data, MathematicalExpression):
            if not isinstance(other.data, pint.Quantity):
                return False
            return self._data.identical(other.data_array)  # type: ignore

        return self._data == other.data

    def __repr__(self):
        """Give __repr__ output."""
        representation = "<TimeSeries>"
        if isinstance(self._data, xr.DataArray):
            if self._data.shape[0] == 1:
                representation += f"\nConstant value:\n\t{self.data.magnitude[0]}\n"
            else:
                representation += (
                    f"\nTime:\n\t{self.time}\n"
                    f"Values:\n\t{self.data.magnitude}\n"
                    f'Interpolation:\n\t{self._data.attrs["interpolation"]}\n'
                )
        else:
            representation += self.data.__repr__().replace(
                "<MathematicalExpression>", ""
            )
        return representation + f"Units:\n\t{self.units}\n"

    @staticmethod
    def _check_data_array(data_array: xr.DataArray):
        """Raise an exception if the 'DataArray' can't be used as 'self._data'."""
        try:
            ut.xr_check_coords(data_array, dict(time={"dtype": ["timedelta64[ns]"]}))
        except (KeyError, TypeError, ValueError) as e:
            raise type(e)(
                "The provided 'DataArray' does not match the required pattern. It "
                "needs to have a dimension called 'time' with coordinates of type "
                "'timedelta64[ns]'. The error reported by the comparison function was:"
                f"\n{e}"
            )

        if not isinstance(data_array.data, pint.Quantity):
            raise TypeError("The data of the 'DataArray' must be a 'pint.Quantity'.")

    @staticmethod
    def _create_data_array(
        data: Union[pint.Quantity, xr.DataArray], time: Time
    ) -> xr.DataArray:
        if isinstance(data, xr.DataArray):
            return data
        return (
            xr.DataArray(data=data)
            .rename({"dim_0": "time"})
            .assign_coords({"time": time.as_timedelta_index()})
        )

    def _initialize_discrete(
        self,
        data: Union[pint.Quantity, xr.DataArray],
        time: types_time_like = None,
        interpolation: str = None,
    ):
        """Initialize the internal data with discrete values."""
        # set default interpolation
        if interpolation is None:
            interpolation = "step"

        if isinstance(data, xr.DataArray):
            self._check_data_array(data)
            data = data.transpose("time", ...)
            self._data = data
            # todo: set _reference_time?
        else:
            # expand dim for scalar input
            data = Q_(data)
            if not np.iterable(data):
                data = np.expand_dims(data, 0)

            # constant value case
            if time is None:
                time = pd.Timedelta(0)
            time = Time(time)

            self._reference_time = time.reference_time
            self._data = self._create_data_array(data, time)
        self.interpolation = interpolation

    def _init_expression(self, data):
        """Initialize the internal data with a mathematical expression."""
        if data.num_variables != 1:
            raise Exception(
                "The mathematical expression must have exactly 1 free "
                "variable that represents time."
            )

        # check that the expression can be evaluated with a time quantity
        time_var_name = data.get_variable_names()[0]
        try:
            eval_data = data.evaluate(**{time_var_name: Q_(1, "second")}).data
            self._units = eval_data.units
            if np.iterable(eval_data):
                self._shape = eval_data.shape
            else:
                self._shape = (1,)
        except pint.errors.DimensionalityError:
            raise Exception(
                "Expression can not be evaluated with "
                '"weldx.Quantity(1, "seconds")"'
                ". Ensure that every parameter posses the correct unit."
            )

        # assign internal variables
        self._data = data
        self._time_var_name = time_var_name

        # check that all parameters of the expression support time arrays
        try:
            self.interp_time(Q_([1, 2], "second"))
            self.interp_time(Q_([1, 2, 3], "second"))
        except Exception as e:
            raise Exception(
                "The expression can not be evaluated with arrays of time deltas. "
                "Ensure that all parameters that are multiplied with the time "
                "variable have an outer dimension of size 1. This dimension is "
                "broadcasted during multiplication. The original error message was:"
                f' "{str(e)}"'
            )

    def _interp_time_discrete(self, time: Time) -> xr.DataArray:
        """Interpolate the time series if its data is composed of discrete values."""
        return ut.xr_interp_like(
            self._data,
            {"time": time.as_data_array()},
            method=self.interpolation,
            assume_sorted=False,
            broadcast_missing=False,
        )

    def _interp_time_expression(self, time: Time, time_unit: str) -> xr.DataArray:
        """Interpolate the time series if its data is a mathematical expression."""
        time_q = time.as_quantity(unit=time_unit)
        if len(time_q.shape) == 0:
            time_q = np.expand_dims(time_q, 0)  # type: ignore[assignment]

        time_xr = xr.DataArray(time_q, dims=["time"])

        # evaluate expression
        data = self._data.evaluate(**{self._time_var_name: time_xr})
        return data.assign_coords({"time": time.as_data_array()})

    @property
    def data(self) -> Union[pint.Quantity, MathematicalExpression]:
        """Return the data of the TimeSeries.

        This is either a set of discrete values/quantities or a mathematical expression.

        Returns
        -------
        pint.Quantity:
            Underlying data array.
        MathematicalExpression:
            A mathematical expression describing the time dependency

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.data
        return self._data

    @property
    def data_array(self) -> Union[xr.DataArray, None]:
        """Return the internal data as 'xarray.DataArray'.

        If the TimeSeries contains an expression, 'None' is returned.

        Returns
        -------
        xarray.DataArray:
            The internal data as 'xarray.DataArray'

        """
        if isinstance(self._data, xr.DataArray):
            return self._data
        return None

    @property
    def interpolation(self) -> Union[str, None]:
        """Return the interpolation.

        Returns
        -------
        str:
            Interpolation of the TimeSeries

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.attrs["interpolation"]
        return None

    @interpolation.setter
    def interpolation(self, interpolation):
        if isinstance(self._data, xr.DataArray):
            if interpolation not in self._valid_interpolations:
                raise ValueError(
                    "A valid interpolation method must be specified if discrete "
                    f'values are used. "{interpolation}" is not supported'
                )
            if self.time is None and interpolation != "step":
                interpolation = "step"
            self.data_array.attrs["interpolation"] = interpolation

    @property
    def is_discrete(self) -> bool:
        """Return `True` if the time series is described by discrete values."""
        return not self.is_expression

    @property
    def is_expression(self) -> bool:
        """Return `True` if the time series is described by an expression."""
        return isinstance(self.data, MathematicalExpression)

    @property
    def time(self) -> Union[None, Time]:
        """Return the data's timestamps.

        Returns
        -------
        pandas.TimedeltaIndex:
            Timestamps of the  data

        """
        if isinstance(self._data, xr.DataArray) and len(self._data.time) > 1:
            return Time(self._data.time.data, self.reference_time)
        return None

    @property
    def reference_time(self) -> Union[pd.Timestamp, None]:
        """Get the reference time."""
        return self._reference_time

    def interp_time(
        self, time: Union[pd.TimedeltaIndex, pint.Quantity, Time], time_unit: str = "s"
    ) -> TimeSeries:
        """Interpolate the TimeSeries in time.

        If the internal data consists of discrete values, an interpolation with the
        prescribed interpolation method is performed. In case of mathematical
        expression, the expression is evaluated for the given timestamps.

        Parameters
        ----------
        time:
            The time values to be used for interpolation.
        time_unit:
            Only important if the time series is described by an expression and a
            'pandas.TimedeltaIndex' is passed to this function. In this case, time is
            converted to a quantity with the provided unit. Even though pint handles
            unit prefixes automatically, the accuracy of the results can be heavily
            influenced if the provided unit results in extreme large or
            small values when compared to the parameters of the expression.

        Returns
        -------
        TimeSeries :
            A new `TimeSeries` object containing the interpolated data.

        """
        if self._interp_counter > 0:
            warn(
                "The data of the time series has already been interpolated "
                f"{self._interp_counter} time(s)."
            )

        # prepare timedelta values for internal interpolation
        time = Time(time)
        time_interp = Time(time, self.reference_time)

        if isinstance(self._data, xr.DataArray):
            dax = self._interp_time_discrete(time_interp)
            ts = TimeSeries(data=dax.data, time=time, interpolation=self.interpolation)
        else:
            dax = self._interp_time_expression(time_interp, time_unit)
            ts = TimeSeries(data=dax, interpolation=self.interpolation)

        ts._interp_counter = self._interp_counter + 1
        return ts

    @check_matplotlib_available
    def plot(
        self,
        time: Union[pd.TimedeltaIndex, pint.Quantity] = None,
        axes: "matplotlib.axes.Axes" = None,  # noqa: F821
        data_name: str = "values",
        time_unit: UnitLike = None,
        **mpl_kwargs,
    ) -> "matplotlib.axes.Axes":  # noqa: F821
        """Plot the `TimeSeries`.

        Parameters
        ----------
        time :
            The points in time that should be plotted. This is an optional parameter for
            discrete `TimeSeries` but mandatory for expression based TimeSeries.
        axes :
            An optional matplotlib axes object
        data_name :
            Name of the data that will appear in the y-axis label
        mpl_kwargs :
            Key word arguments that are passed to the matplotlib plot function
        time_unit :
            The desired time unit for the plot. If `None` is provided, the internally
            stored unit will be used.

        Returns
        -------
         matplotlib.axes.Axes :
            The matplotlib axes object that was used for the plot

        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots()
        if self.is_expression or time is not None:
            return self.interp_time(time).plot(
                axes=axes, data_name=data_name, time_unit=time_unit, **mpl_kwargs
            )

        time = Time(self.time, self.reference_time).as_quantity()
        if time_unit is not None:
            time = time.to(time_unit)

        axes.plot(time.m, self._data.data.m, **mpl_kwargs)  # type: ignore
        axes.set_xlabel(f"t in {time.u:~}")
        y_unit_label = ""
        if self.units not in ["", "dimensionless"]:
            y_unit_label = f" in {self.units:~}"
        axes.set_ylabel(data_name + y_unit_label)

        return axes

    @property
    def shape(self) -> tuple:
        """Return the shape of the TimeSeries data.

        For mathematical expressions, the shape does not contain the time axis.

        Returns
        -------
        Tuple:
            Tuple describing the data's shape

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.shape
        return self._shape

    @property
    def units(self) -> pint.Unit:
        """Return the units of the TimeSeries Data.

        Returns
        -------
        pint.Unit:
            The unit of the `TimeSeries`

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.data.units
        return self._units


# GenericSeries ------------------------------------------------------------------------

# todo
#  - pandas time types in TimeSeries vs GenericSeries
#
#  - add doctests (examples)


def _quantity_to_coord_tuple(
    v: pint.Quantity, dim
) -> tuple[str, np.ndarray, dict[str, pint.Unit]]:
    return (dim, v.m, {UNITS_KEY: v.u})


def _quantity_to_xarray(v: pint.Quantity, dim: str = None) -> xr.DataArray:
    """Convert a single quantity into a formatted xarray dataarray."""
    return xr.DataArray(v, dims=dim)


def _quantities_to_xarray(q_dict: dict[str, pint.Quantity]) -> dict[str, xr.DataArray]:
    """Convert a str:Quantity mapping into a mapping of `xarray.DataArray`."""
    return {k: _quantity_to_xarray(v, k) for k, v in q_dict.items()}


@dataclass
class SeriesParameter:
    """Describes a parameter/coordinate of a Series and convert between formats.

    The input value gets stored as either quantity or DataArray.
    (DataArray is stored 'as is', other inputs will be converted to quantities).

    In addition, the desired dimension on the Parameter and an optional symbol
    representation for math expressions can be added.

    The stored value can be converted to different formats available as properties.
    """

    values: Union[xr.DataArray, pint.Quantity]
    """The values of the parameter are stored as quantities or DataArrays"""
    dim: str = None
    """The xarray dimension associated with the parameter."""
    symbol: str = None
    """The math expression symbol associated with the parameter."""

    def __post_init__(self):
        """Convert inputs and validate values."""
        if isinstance(self.values, SeriesParameter):
            self.dim = self.values.dim
            self.symbol = self.values.symbol
            self.values = self.values.values
            return

        if isinstance(self.values, tuple):
            self.dim = self.values[1]
            self.values = Q_(self.values[0])

        if not isinstance(self.values, (pint.Quantity, xr.DataArray)):
            self.values = Q_(self.values)

        if not isinstance(self.values, (pint.Quantity, xr.DataArray)):
            raise ValueError(f"Cannot set parameter as {self.values}")

    @property
    def units(self) -> pint.Unit:
        """Get the units information of the parameter."""
        if isinstance(self.values, pint.Quantity):
            return self.values.units
        return self.values.weldx.units

    @property
    def data_array(self) -> xr.DataArray:
        """Get the parameter formatted as xarray."""
        if isinstance(self.values, xr.DataArray):
            return self.values
        # we cannot have scalar values here
        values = self.values
        if not values.shape:
            values = np.expand_dims(values, 0)
        return _quantity_to_xarray(values, self.dim)  # type: ignore[arg-type]

    @property
    def quantity(self) -> pint.Quantity:
        """Get the parameter formatted as a quantity."""
        if isinstance(self.values, pint.Quantity):
            return self.values
        return self.values.weldx.quantify().data

    @property
    def coord_tuple(self) -> tuple[str, np.ndarray, dict[str, pint.Unit]]:
        """Get the parameter in xarray coordinate tuple format."""
        if isinstance(self.values, pint.Quantity):
            return _quantity_to_coord_tuple(self.values, self.dim)
        da: xr.DataArray = self.values.pint.dequantify()
        return self.dim, da.data, da.weldx.units


class GenericSeries:
    """Describes a quantity depending on one or more parameters."""

    _allowed_variables: list[str] = []
    """A list of allowed variable names. (only expression)

    If the expression contains any other variable name that is not part of the list,
    an exception is raised. It is not required that an expression includes all these
    variables. Additionally, the expression can contain other symbols if they are used
    as parameters.
    """
    _required_variables: list[str] = []
    """A list of required variable names. (only expression)

    If one or more variables are missing in the expression, an exceptions is raised.
    Note that the required symbols must be variables of the expression. Using one or
    more as a parameter will also trigger an exception.
    """

    _evaluation_preprocessor: dict[str, Callable] = {}
    """Mapping of variable names to functions that are applied prior to evaluation.

    When calling `GenericSeries.evaluate`, the passed keyword arguments are checked
    against the dictionaries keys. If a match is found, the corresponding preprocessor
    function is called with the variables value and returns the updated value. As an
    example, this can be used to support multiple time formats. The key might be ``t``
    and the preprocessor function would turn the original time data into an equivalent
    `xarray.DataArray`.
    """

    _required_dimensions: list[str] = []
    """A list of required dimension names.

    Explicit `GenericSeries` need all of the listed dimensions. Otherwise an exception
    is raised. If the series is based on an expression, the dimension can either be
    represented by a variable or be part of one of the expressions parameters.
    """

    _required_dimension_units: dict[str, pint.Unit] = {}
    """A dictionary that maps a required unit dimensionality to a dimension.

    If a dimension matches one of the keys of this dictionary, its dimensionality
    is checked against the listed requirement.
    """
    _required_dimension_coordinates: dict[str, list] = {}
    """A dictionary that maps required coordinate values to a dimension.

    If a dimension matches one of the keys of this dictionary, it is checked if it has
    the specified coordinate values. An example use-case would be a 3d-space where the
    coordinates "x", "y" and "z" are required for a spatial dimension.
    """

    _required_unit_dimensionality: pint.Unit = None
    """Required unit dimensionality of the evaluated expression/data.

    If the defined unit does not result from the evaluation of the series, an exception
    is raised. Note that this already checked during construction. If `None`, no
    specific unit is required. A unit-less series can be enforced by setting this
    setup variable to ``""``.
    """

    # do it later

    _allowed_dimensions: list[str] = NotImplemented
    """A list of allowed dimension names."""
    _required_parameter_shape: dict[str, int] = NotImplemented
    """Size of the parameter dimensions/coordinates - (also defines parameter order)"""
    _alias_names: dict[str, list[str]] = NotImplemented
    """Allowed alias names for a variable or parameter in an expression"""

    def __init__(
        self,
        obj: Union[pint.Quantity, xr.DataArray, str, MathematicalExpression],
        dims: Union[list[str], dict[str, str]] = None,
        coords: dict[str, Union[list, pint.Quantity]] = None,
        units: dict[str, Union[str, pint.Unit]] = None,
        interpolation: str = None,
        parameters: dict[str, Union[str, pint.Quantity, xr.DataArray]] = None,
    ):
        """Create a generic series.

        Parameters
        ----------
        obj :
            Either a multidimensional array of discrete values or a
            `MathematicalExpression` with one or more variables. The expression can also
            be provided as string. In this case, you need to provide all parameters
            using the corresponding interface variable (see below).
        dims :
            For discrete data, a list is expected that provides the dimension names.
            The first name refers to the outer most dimension. If an expression is used,
            this parameter is optional. It can be used to have a dimension name that
            differs from the symbol of the expression. To do so, you need to provide a
            mapping between the symbol and the dimension name. For example, you could
            use ``dict(t="time")`` to tell the `GenericSeries` that the symbol ``t``
            refers to the dimension ``time``.
        coords :
            (Only for discrete values) A mapping that specifies the coordinate values
            for each dimension.
        units :
            (Only for expressions) A mapping that specifies the expected unit for a
            free dimension/expression variable. During evaluation, it is not necessary
            that the provided data points have the exact same unit, but it must be a
            compatible unit. For example, if we use ``dict(t="s")`` we can use minutes,
            hours or any other time unit for ``t`` during evaluation, but using meters
            would cause an error.
        interpolation :
            (Only for discrete values) The interpolating method that should be used
            during evaluation.
        parameters :
            (Only for expressions) Parameters to set in the math expression.

        Raises
        ------
        TypeError
            If ``obj`` is any other type than the ones defined in the type hints.
        KeyError
            If one of the provided mappings refers to a symbol that is not part of the
            expression
        ValueError
            Can be raised for multiple reasons related to incompatible or invalid values
        pint.DimensionalityError
            If an expression can not be evaluated due to a unit conflict caused by
            the provided parameters and and dimension units

        Examples
        --------
        Create a `GenericSeries` representing a translation with 3 m/s in x-direction
        starting at point ``[0, 2 ,2] cm``

        >>> from weldx import GenericSeries, Q_
        >>> GenericSeries(  # doctest: +NORMALIZE_WHITESPACE
        ...     "a*t + b",
        ...     parameters=dict(a=Q_([3, 0, 0], "mm/s"), b=Q_([0, 2, 2], "cm")),
        ...     units=dict(t="s"),
        ... )
        <GenericSeries>
        Expression:
            a*t + b
        Parameters:
            a = [3 0 0] mm / s
            b = [0 2 2] cm
        Free Dimensions:
            t in s
        Other Dimensions:
            ['dim_0']
        Units:
            mm


        The same `GenericSeries` from above but assigning the ``t`` parameter to the
        output dimension ``time``.

        >>> GenericSeries(  # doctest: +NORMALIZE_WHITESPACE
        ...     "a*t + b",
        ...     parameters=dict(a=Q_([3, 0, 0], "mm/s"), b=Q_([0, 2, 2], "cm")),
        ...     units=dict(t="s"),
        ...     dims=dict(t="time"),
        ... )
        <GenericSeries>
        Expression:
            a*t + b
        Parameters:
            a = [3 0 0] mm / s
            b = [0 2 2] cm
        Free Dimensions:
            t in s
        Other Dimensions:
            ['dim_0']
        Units:
            mm

        A `GenericSeries` describing linear interpolation between the values 10 V and
        20 V over a period of 5 seconds.

        >>> GenericSeries(  # doctest: +NORMALIZE_WHITESPACE
        ...     Q_([10, 20], "V"),
        ...     dims=["t"],
        ...     coords={"t": Q_([0, 5], "s")},
        ... )
        <GenericSeries>
        Values:
            [10 20]
        Dimensions:
            ('t',)
        Coordinates:
            t      = [0 5] s
        Units:
            V


        """
        if units is None:
            units = {}

        self._obj: Union[xr.DataArray, MathematicalExpression] = None
        self._variable_units: dict[str, pint.Unit] = None
        self._symbol_dims: bidict = bidict({})
        self._units: pint.Unit = None
        self._interpolation = "linear" if interpolation is None else interpolation

        if isinstance(obj, (pint.Quantity, xr.DataArray)):
            if dims is not None and not isinstance(dims, list):
                raise ValueError(f"Argument 'dims' must be list of strings, not {dims}")
            self._init_discrete(obj, dims, coords)
        elif isinstance(obj, (MathematicalExpression, str, sympy.Expr)):
            if dims is not None and not isinstance(dims, dict):
                raise ValueError(f"Argument 'dims' must be dict, not {dims}")
            self._init_expression(
                obj, dims, parameters, {k: U_(v) for k, v in units.items()}  # catch str
            )
        else:
            raise TypeError(f'The data type "{type(obj)}" is not supported.')

    def __eq__(self, other):
        """Compare the Generic Series to another object."""
        from weldx.util import compare_nested

        # todo: what about derived GS types? Maybe add another is_equivalent function?
        if not isinstance(other, type(self)):
            return False

        if self.is_expression != other.is_expression:
            return False

        if self.is_expression:
            if not compare_nested(self._symbol_dims, other._symbol_dims):
                return False
            if not compare_nested(self._variable_units, other._variable_units):
                return False
            return self._obj == other._obj

        if self.interpolation != other.interpolation:
            return False
        return self.data_array.identical(other._obj)

    def _init_discrete(
        self,
        data: Union[pint.Quantity, xr.DataArray],
        dims: list[str],
        coords: dict[str, Union[list, pint.Quantity]],
    ):
        """Initialize the internal data with discrete values."""
        if not isinstance(data, xr.DataArray):
            if coords is not None:
                coords = {
                    k: SeriesParameter(v, k).coord_tuple  # type: ignore[misc]
                    for k, v in coords.items()
                }
            data = xr.DataArray(data=data, dims=dims, coords=coords).weldx.quantify()
        else:
            # todo check data structure
            pass
        # check the constraints of derived types
        self._check_constraints_discrete(data)
        self._obj = data

    @staticmethod
    def _init_get_updated_dims(
        expr: MathematicalExpression, dims: dict[str, str] = None
    ) -> dict[str, str]:
        if dims is None:
            dims = {}
        return {v: dims.get(v, v) for v in expr.get_variable_names()}

    def _init_get_updated_units(
        self,
        expr: MathematicalExpression,
        units: dict[str, pint.Unit],
    ) -> dict[str, pint.Unit]:
        """Cast dimensions and units into the internally used, unified format."""
        if units is None:
            units = {}

        if self._required_dimension_units is not None:
            for k, v in self._required_dimension_units.items():
                if k not in units and k not in expr.parameters:
                    units[k] = v
        for k2, v2 in units.items():
            if k2 not in expr.get_variable_names():
                raise KeyError(f"{k2} is not a variable of the expression:\n{expr}")
            units[k2] = U_(v2)

        for val in expr.get_variable_names():
            if val not in units:
                units[val] = U_("")

        return units

    def _init_expression(
        self,
        expr: Union[str, MathematicalExpression],
        dims: dict[str, str],
        parameters: dict[str, Union[str, pint.Quantity, xr.DataArray]],
        units: dict[str, pint.Unit],
    ):
        """Initialize the internal data with a mathematical expression."""
        # Check and update expression
        if isinstance(expr, MathematicalExpression):
            parameters = expr.parameters
            expr = str(expr.expression)
        if parameters is not None:
            parameters = self._format_expression_params(parameters)
        expr = MathematicalExpression(expr, parameters)

        if expr.num_variables == 0:
            raise ValueError("The passed expression has no variables.")

        # Update units and dims
        dims = self._init_get_updated_dims(expr, dims)
        units = self._init_get_updated_units(expr, units)

        # check expression
        expr_units = self._test_expr(expr, dims, units)

        # check constraints
        self._check_constraints_expression(expr, dims, units, expr_units)

        # save internal data
        self._units = expr_units
        self._obj = expr
        self._variable_units = units
        self._symbol_dims = bidict(dims)

    @staticmethod
    def _test_expr(expr, dims, units: dict[str, pint.Unit]) -> pint.Unit:
        """Perform a test evaluation of the expression to determine the resulting units.

        This function assures that all of the provided information are compatible
        (units, array lengths, etc.). It also determines the output unit of the
        expression.
        """
        try:
            scalar_params = {k: Q_(1, v) for k, v in units.items()}
            result = expr.evaluate(**scalar_params)
            expr_units = result.data.to_reduced_units().units
        except pint.errors.DimensionalityError as e:
            raise pint.DimensionalityError(
                e.units1,
                e.units2,
                extra_msg="\nExpression can not be evaluated due to a unit "
                "dimensionality error. Ensure that the expressions parameters and the "
                "expected variable units are compatible. The original exception was:\n"
                f"{e}",
            )
        except ValueError:
            pass  # Error message will be generated by the next check

        try:
            # we evaluate twice with different array sizes because it might happen that
            # a parameter uses the same dimension as a variable but the check still
            # passes because the test input for the variable has the same array length.
            # This case will be caught in the second evaluation.
            for offset in range(2):
                array_params = {
                    k: xr.DataArray(Q_(range(i + 2 + offset), v), dims=dims[k])
                    for i, (k, v) in enumerate(units.items())
                }
                expr.evaluate(**array_params)
        except ValueError as e:
            raise ValueError(
                "During the evaluation of the expression mismatching array lengths' "
                "were detected. Some possible causes are:\n"
                "  - expression parameters that have already assigned coordinates to a "
                "dimension that is also used as a variable\n"
                "  - 2 free dimensions with identical names\n"
                "  - 2 expression parameters that use the same dimension with "
                "different number of values\n"
                f"The original exception was:\n{e}"
            )

        return expr_units

    @staticmethod
    def _format_expression_params(
        parameters: dict[str, Union[pint.Quantity, xr.DataArray]]
    ) -> dict[str, Union[pint.Quantity, xr.DataArray]]:
        """Create expression parameters as a valid internal type.

        Valid types are all input types for the `MathematicalExpression`, with the
        limitation that every parameter needs a unit.
        """
        # todo
        #  - enable usage of dicts for params (data, dims, coords)
        #  - tuple should accept third element (coords)

        params = [SeriesParameter(v, symbol=k) for k, v in parameters.items()]
        for v in params:
            if v.units is None:
                raise ValueError(f"Value for parameter {v} is not a quantity.")

        return {
            p.symbol: (
                (p.values, p.dim)  # need to preserve tuple format (quantity, dim)
                if isinstance(p.values, pint.Quantity) and p.dim != p.symbol
                else p.values
            )
            for p in params
        }

    def __repr__(self):
        """Give __repr__ output."""
        # todo: remove scalar dims?
        rep = f"<{type(self).__name__}>\n"
        if self.is_discrete:
            arr_str = np.array2string(
                self._obj.data.magnitude, threshold=3, precision=4, prefix="        "
            )
            rep += f"Values:\n\t{arr_str}\n"
            rep += f"Dimensions:\n\t{self.dims}\n"
            rep += "Coordinates:\n"
            for coord, val in self.coordinates.items():
                c_d = np.array2string(val.data, threshold=3, precision=4)
                rep += f"\t{coord}".ljust(7)
                rep += f" = {c_d}"
                rep += f" {val.attrs.get(UNITS_KEY)}\n"
        else:
            rep += self.data.__repr__().replace("<MathematicalExpression>\n", "")
            rep += "Free Dimensions:\n"
            for k, v in self._variable_units.items():
                rep += f"\t{k} in {v}\n"
            rep += "Other Dimensions:\n"
            _variable_units_replaced = {
                self._symbol_dims.get(k, k): u for k, u in self._variable_units.items()
            }
            rep += f"\t{[v for v in self.dims if v not in _variable_units_replaced]}\n"

        return rep + f"Units:\n\t{self.units}\n"

    # evaluate functions ---------------------------------------------

    def evaluate(self, **kwargs) -> GenericSeries:
        """Evaluate the generic series at discrete coordinates.

        If the `GenericSeries` is composed of discrete values, the data is interpolated
        using the specified interpolation method.

        Expressions are simply evaluated if coordinates for all dimensions are provided
        which results in a new discrete `GenericSeries`. In case that some dimensions
        are left without coordinates, a new expression based `GenericSeries` is
        returned. The provided coordinates are stored as parameters and the
        corresponding dimensions are no longer variables of the new `GenericSeries`.

        Parameters
        ----------
        kwargs:
            An arbitrary number of keyword arguments. The key must be a dimension name
            of the `GenericSeries` and the values are the corresponding coordinates
            where the `GenericSeries` should be evaluated. It is not necessary to
            provide values for all dimensions. Partial evaluation is also possible.

        Returns
        -------
        GenericSeries :
            A new generic series with the (partially) evaluated data.

        """
        coords = self._evaluate_preprocessor(**kwargs)

        if self.is_expression:
            return self._evaluate_expr(coords)
        return self._evaluate_array(coords)

    def _evaluate_preprocessor(self, **kwargs) -> list[SeriesParameter]:
        """Preprocess the passed parameters into coordinates for evaluation."""
        kwargs = ut.apply_func_by_mapping(
            self.__class__._evaluation_preprocessor,  # type: ignore # skipcq: PYL-W0212
            kwargs,
        )

        coords = [
            SeriesParameter(v, k, symbol=self._symbol_dims.inverse.get(k, k))
            for k, v in kwargs.items()
        ]

        return coords

    def _evaluate_expr(self, coords: list[SeriesParameter]) -> GenericSeries:
        """Evaluate the expression at the passed coordinates."""
        if len(coords) == self._obj.num_variables:
            eval_args = {
                v.symbol: v.data_array.assign_coords(
                    {v.dim: v.data_array.pint.dequantify()}
                )
                for v in coords
            }
            da = self._obj.evaluate(**eval_args)
            return self.__class__(da)

        # turn passed coords into parameters of the expression
        new_series = deepcopy(self)
        for p in coords:
            new_series._obj.set_parameter(  # skipcq: PYL-W0212
                p.symbol, (p.quantity, p.dim)
            )
            new_series._symbol_dims.pop(p.symbol)  # skipcq: PYL-W0212
            new_series._variable_units.pop(p.symbol)  # skipcq: PYL-W0212
        return new_series

    def _evaluate_array(self, coords: list[SeriesParameter]) -> GenericSeries:
        """Evaluate (interpolate) discrete Series object at the coordinates."""
        eval_args = {v.dim: v.data_array.pint.dequantify() for v in coords}
        for k in eval_args:
            if k not in self.data_array.dims:
                raise KeyError(f"'{k}' is not a valid dimension.")
        return self.__class__(
            ut.xr_interp_like(self._obj, da2=eval_args, method=self._interpolation)
        )

    def __call__(self, **kwargs) -> GenericSeries:
        """Evaluate the generic series at discrete coordinates.

        For a detailed description read the documentation of the`evaluate` function.

        """
        return self.evaluate(**kwargs)

    # properties etc. ---------------------------------------------

    @property
    def coordinates(self) -> Union[DataArrayCoordinates, None]:
        """Get the coordinates of the generic series."""
        if self.is_discrete:
            return self.data_array.coords
        return None

    @property
    def coordinate_names(self) -> list[str]:
        """Get the names of all coordinates."""
        return NotImplemented

    @property
    def data(self) -> Union[pint.Quantity, MathematicalExpression]:
        """Get the internal data."""
        if self.is_discrete:
            return self.data_array.data
        return self._obj

    @property
    def data_array(self) -> Union[xr.DataArray, None]:
        """Get the internal data as `xarray.DataArray`."""
        if self.is_discrete:
            return self._obj
        return None

    @staticmethod
    def _get_expression_dims(
        expr: MathematicalExpression, symbol_dims: Mapping[str, str]
    ) -> list[str]:
        """Get the dimensions of an expression based `GenericSeries`.

        This is the union of parameter dimensions and free dimensions.

        """
        dims = set(symbol_dims.values())
        for v in expr.parameters.values():
            if not isinstance(v, xr.DataArray):
                v = xr.DataArray(v)
            if v.size > 0:
                dims |= set(v.dims)
        return list(dims)

    @property
    def dims(self) -> list[str]:
        """Get the names of all dimensions."""
        if self.is_expression:
            return self._get_expression_dims(self._obj, self._symbol_dims)
        return self.data_array.dims

    @property
    def interpolation(self) -> str:
        """Get the name of the used interpolation method."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, val: str):
        if val not in [
            "linear",
            "step",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
        ]:
            raise ValueError(f"'{val}' is not a supported interpolation method.")
        self._interpolation = val

    @property
    def is_discrete(self) -> bool:
        """Return `True` if the time series is described by discrete values."""
        return isinstance(self._obj, xr.DataArray)

    @property
    def is_expression(self) -> bool:
        """Return `True` if the time series is described by an expression."""
        return isinstance(self._obj, MathematicalExpression)

    @property
    def ndims(self) -> int:
        """Get the number of dimensions."""
        return len(self.dims)

    @property
    def variable_names(self) -> Union[list[str], None]:
        """Get the names of all variables."""
        if self.is_expression:
            return list(self._variable_units.keys())
        return None

    @property
    def variable_units(self) -> dict[str, pint.Unit]:
        """Get a dictionary that maps the variable names to their expected units."""
        return self._variable_units

    @property
    def shape(self) -> tuple:
        """Get the shape of the generic series data."""
        if self.is_expression:
            return NotImplemented
        return self.data_array.shape
        # todo Expression? -> dict shape?

    @property
    def units(self) -> pint.Unit:
        """Get the units of the generic series data."""
        if self.is_expression:
            return self._units
        return self.data_array.pint.units

    # constraint checks for derived series ---------------------------------------------

    @classmethod
    def _check_req_items(cls, req, data, desc):
        """Check if all required items are contained in `data`."""
        if not set(req).issubset(data):
            raise ValueError(f"{cls.__name__} requires {desc} '{req}'.")

    @classmethod
    def _check_constraints_discrete(cls, data_array: xr.DataArray):
        """Check if the constraints of a discrete derived type are met."""
        if cls is GenericSeries:
            return

        # check dimension constraints
        cls._check_req_items(cls._required_dimensions, data_array.dims, "dimensions")

        # check dimensionality constraint
        ut.xr_check_dimensionality(data_array, cls._required_unit_dimensionality)

        # check coordinate constraints
        _units = cls._required_dimension_units
        _vals = cls._required_dimension_coordinates
        _keys = (set(_units.keys()) & set(data_array.dims)) | set(_vals.keys())

        ref: dict[str, dict] = {k: {} for k in _keys}
        for k in ref.keys():
            if k in _units:
                ref[k]["dimensionality"] = _units[k]
            if k in _vals:
                ref[k]["values"] = _vals[k]
        ut.xr_check_coords(data_array, ref)

    @classmethod
    def _check_constraints_expression(
        cls,
        expr: MathematicalExpression,
        var_dims: dict[str, str],
        var_units: dict[str, pint.Unit],
        expr_units: pint.Unit,
    ):
        """Check if the constraints of an expression based derived type are met."""
        if cls is GenericSeries:
            return

        # check variable constraints
        var_names = expr.get_variable_names()
        vars_allow = cls._allowed_variables

        if len(vars_allow) > 0 and not set(var_names).issubset(vars_allow):
            raise ValueError(
                f"'{var_names}' is not a subset of the allowed expression variables "
                f"{vars_allow} of class {cls.__name__}"
            )
        cls._check_req_items(cls._required_variables, var_names, "expression variables")

        # check dimension constraints
        cls._check_req_items(
            cls._required_dimensions,
            cls._get_expression_dims(expr, var_dims),
            "dimensions",
        )

        # check dimensionality constraint
        req_dimty = cls._required_unit_dimensionality

        if req_dimty is not None and not expr_units.is_compatible_with(req_dimty):
            raise pint.DimensionalityError(
                expr_units,
                req_dimty,
                extra_msg=f"\n{cls.__name__} requires its output unit to be of "
                f"dimensionality '{req_dimty.dimensionality}' but it actually is "
                f"'{expr_units.dimensionality}'.",
            )

        # check units of dimensions
        for k, v in cls._required_dimension_units.items():
            d_units = var_units.get(k)
            param = expr.parameters.get(k)

            if d_units is None and param is not None:
                d_units = param.u if isinstance(param, pint.Quantity) else param.data.u

            if d_units is None or not U_(d_units).is_compatible_with(U_(v)):
                raise pint.DimensionalityError(
                    U_(v),
                    U_(d_units),
                    extra_msg=f"\n{cls.__name__} requires dimension {k} to have the "
                    f"unit dimensionality '{U_(v).dimensionality}'",
                )

        # check coords
        for k, v in cls._required_dimension_coordinates.items():
            if k in var_dims:
                raise ValueError(
                    f"{cls.__name__} requires dimension {k} to have the "
                    f"coordinates {v}. Therefore it can't be a variable dimension."
                )

            ref = {k: {"values": v}}
            for param in expr.parameters.values():
                if isinstance(param, xr.DataArray) and k in param.coords.keys():
                    ut.xr_check_coords(param, ref)

            # todo: add limits for dims?

    # not yet implemented ---------------------------------------------

    def __add__(self, other):
        """Add two `GenericSeries`."""
        # this should mostly be moved to the MathematicalExpression
        # todo:
        #   - for two expressions simply do: f"{exp_1} + f{exp_2}" and merge the
        #     parameters in a new MathExpression
        #   - for two discrete series call __add__ of the xarrays
        #   - for mixed version add a new parameter to the expression string and set the
        #     xarray as the parameters value
        return NotImplemented

    @staticmethod
    # skipcq: PYL-W0613
    def interp_like(
        obj: Any, dimensions: list[str] = None, accessor_mappings: dict = None
    ) -> GenericSeries:
        """Interpolate using the coordinates of another object.

        Parameters
        ----------
        obj :
            An object that provides the coordinate values.
        dimensions :
            The dimensions that should be interpolated. If `None` is passed, all
            dimensions will be interpolated
        accessor_mappings :
            A mapping between the dimensions of the generic series and the corresponding
            coordinate accessor of the provided object. This can be used if the
            coordinate names do not match for the time series and the provided object.

        Returns
        -------
        GenericSeries :
            A new generic series containing discrete values for the interpolated
            dimensions.

        """
        return NotImplemented


# --------------------------------------------------------------------------------------
# SpatialSeries
# --------------------------------------------------------------------------------------


class SpatialSeries(GenericSeries):
    """Describes a line in 3d space depending on the positional coordinate ``s``."""

    _position_dim_name = "s"

    _required_variables: list[str] = [_position_dim_name]
    """Required variable names"""

    _required_dimensions: list[str] = [_position_dim_name, "c"]
    """Required dimensions"""
    _required_dimension_units: dict[str, pint.Unit] = {_position_dim_name: U_("")}
    """Required units of a dimension"""
    _required_dimension_coordinates: dict[str, list] = {"c": ["x", "y", "z"]}
    """Required coordinates of a dimension."""

    def __init__(
        self,
        obj: Union[pint.Quantity, xr.DataArray, str, MathematicalExpression],
        dims: Union[list[str], dict[str, str]] = None,
        coords: dict[str, Union[list, pint.Quantity]] = None,
        units: dict[str, Union[str, pint.Unit]] = None,
        interpolation: str = None,
        parameters: dict[str, Union[str, pint.Quantity, xr.DataArray]] = None,
    ):
        if isinstance(obj, Q_):
            obj = self._process_quantity(obj, dims, coords)
            dims = None
            coords = None
        if parameters is not None:
            parameters = self._process_parameters(parameters)
        super().__init__(obj, dims, coords, units, interpolation, parameters)

    @classmethod
    def _process_quantity(
        cls,
        obj: Union[pint.Quantity, xr.DataArray, str, MathematicalExpression],
        dims: Union[list[str], dict[str, str]],
        coords: dict[str, Union[list, pint.Quantity]],
    ) -> xr.DataArray:
        """Turn a quantity into a a correctly formatted data array."""
        if isinstance(coords, dict):
            s = coords[cls._position_dim_name]
        else:
            s = coords
            coords = {cls._position_dim_name: s}

        if not isinstance(s, xr.DataArray):
            if not isinstance(s, Q_):
                s = Q_(s, "")
            s = xr.DataArray(s, dims=[cls._position_dim_name]).pint.dequantify()
            coords[cls._position_dim_name] = s

        if "c" not in coords:
            coords["c"] = ["x", "y", "z"]

        if dims is None:
            dims = [cls._position_dim_name, "c"]

        return xr.DataArray(obj, dims=dims, coords=coords)

    @staticmethod
    def _process_parameters(params):
        """Turn quantity parameters into the correctly formatted data arrays."""
        for k, v in params.items():
            if isinstance(v, Q_) and v.size == 3:
                params[k] = xr.DataArray(v, dims=["c"], coords=dict(c=["x", "y", "z"]))
        return params

    @property
    def position_dim_name(self):
        """Return the name of the dimension that determines the position on the line."""
        return self._position_dim_name
