"""Collection of common classes and functions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import pint
import xarray as xr

import weldx.util as ut
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG

if TYPE_CHECKING:
    import sympy

__all__ = ["MathematicalExpression", "TimeSeries"]


class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(
        self,
        expression: Union[sympy.Expr, str],
        parameters: Dict[str, Union[pint.Quantity, str]] = None,
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
        import sympy

        if not isinstance(expression, sympy.Expr):
            expression = sympy.sympify(expression)
        self._expression = expression
        self.function = sympy.lambdify(
            tuple(self._expression.free_symbols), self._expression, "numpy"
        )
        self._parameters = {}
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError(
                    f'"parameters" must be dictionary, got {type(parameters)}'
                )
            parameters = {k: Q_(v) for k, v in parameters.items()}
            variable_names = self.get_variable_names()
            for key in parameters:
                if key not in variable_names:
                    raise ValueError(
                        f'The expression does not have a parameter "{key}"'
                    )
            self._parameters = parameters

    def __repr__(self):
        """Give __repr__ output."""
        representation = (
            f"<MathematicalExpression>\n"
            f"Expression:\n\t {self._expression.__repr__()}"
            f"\nParameters:\n"
        )
        if len(self._parameters) > 0:
            for parameter, value in self._parameters.items():
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
        if not isinstance(name, str):
            raise TypeError(f'Parameter "name" must be a string, got {type(name)}')
        if name not in str(self._expression.free_symbols):
            raise ValueError(
                f'The expression "{self._expression}" does not have a '
                f'parameter with name "{name}".'
            )
        self._parameters[name] = value

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
    def parameters(self) -> Dict:
        """Return the internal parameters dictionary.

        Returns
        -------
        Dict
            Internal parameters dictionary

        """
        return self._parameters

    def get_variable_names(self) -> List:
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
        inputs = {**kwargs, **self._parameters}
        return self.function(**inputs)


# TimeSeries ---------------------------------------------------------------------------


class TimeSeries:
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
        time: Union[None, pd.TimedeltaIndex, pint.Quantity] = None,
        interpolation: str = None,
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
        self._data = None
        self._time_var_name = None
        self._shape = None
        self._units = None
        self._interp_counter = 0

        if isinstance(data, pint.Quantity):
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
            return self._data.identical(other.data_array)

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

    def _initialize_discrete(
        self,
        data: pint.Quantity,
        time: Union[None, pd.TimedeltaIndex, pint.Quantity],
        interpolation: str,
    ):
        """Initialize the internal data with discrete values."""
        # set default interpolation
        if interpolation is None:
            interpolation = "step"

        # expand dim for scalar input
        data = Q_(data)
        if not np.iterable(data):
            data = np.expand_dims(data, 0)

        # constant value case
        if time is None:
            time = pd.TimedeltaIndex([0])

        if isinstance(time, pint.Quantity):
            time = ut.to_pandas_time_index(time)
        if not isinstance(time, pd.TimedeltaIndex):
            raise ValueError(
                '"time" must be a time quantity or a "pandas.TimedeltaIndex".'
            )

        dax = xr.DataArray(data=data)
        self._data = dax.rename({"dim_0": "time"}).assign_coords({"time": time})
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
            eval_data = data.evaluate(**{time_var_name: Q_(1, "second")})
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

    def _interp_time_discrete(
        self, time: Union[pd.TimedeltaIndex, pint.Quantity]
    ) -> xr.DataArray:
        """Interpolate the time series if its data is composed of discrete values.

        See `interp_time` for interface description.

        """
        if isinstance(time, pint.Quantity):
            time = ut.to_pandas_time_index(time)
        if not isinstance(time, pd.TimedeltaIndex):
            raise ValueError(
                '"time" must be a time quantity or a "pandas.TimedeltaIndex".'
            )

        return ut.xr_interp_like(
            self._data,
            {"time": time},
            method=self.interpolation,
            assume_sorted=False,
            broadcast_missing=False,
        )

    def _interp_time_expression(
        self, time: Union[pd.TimedeltaIndex, pint.Quantity], time_unit: str
    ) -> xr.DataArray:
        """Interpolate the time series if its data is a mathematical expression.

        See `interp_time` for interface description.

        """
        # Transform time to both formats
        if isinstance(time, pint.Quantity) and time.check(UREG.get_dimensionality("s")):
            time_q = time
            time_pd = ut.to_pandas_time_index(time)
        elif isinstance(time, pd.TimedeltaIndex):
            time_q = ut.pandas_time_delta_to_quantity(time, time_unit)
            time_pd = time
        else:
            raise ValueError(
                '"time" must be a time quantity or a "pandas.TimedeltaIndex".'
            )

        if len(self.shape) > 1 and np.iterable(time_q):
            while len(time_q.shape) < len(self.shape):
                time_q = time_q[:, np.newaxis]

        # evaluate expression
        data = self._data.evaluate(**{self._time_var_name: time_q})
        data = data.astype(float).to_reduced_units()  # float conversion before reduce!

        # create data array
        if not np.iterable(data):  # make sure quantity is not scalar value
            data = np.expand_dims(data, 0)

        dax = xr.DataArray(data=data)  # don't know exact dimensions so far
        return dax.rename({"dim_0": "time"}).assign_coords({"time": time_pd})

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
    def time(self) -> Union[None, pd.TimedeltaIndex]:
        """Return the data's timestamps.

        Returns
        -------
        pandas.TimedeltaIndex:
            Timestamps of the  data

        """
        if isinstance(self._data, xr.DataArray) and len(self._data.time) > 1:
            return ut.to_pandas_time_index(self._data.time.data)
        return None

    def interp_time(
        self, time: Union[pd.TimedeltaIndex, pint.Quantity], time_unit: str = "s"
    ) -> "TimeSeries":
        """Interpolate the TimeSeries in time.

        If the internal data consists of discrete values, an interpolation with the
        prescribed interpolation method is performed. In case of mathematical
        expression, the expression is evaluated for the given timestamps.

        Parameters
        ----------
        time:
            A set of timestamps.
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

        if isinstance(self._data, xr.DataArray):
            dax = self._interp_time_discrete(time)
        else:
            dax = self._interp_time_expression(time, time_unit)

        ts = TimeSeries(data=dax.data, time=time, interpolation=self.interpolation)
        ts._interp_counter = self._interp_counter + 1
        return ts

    @property
    def shape(self) -> Tuple:
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
    def units(self) -> str:
        """Return the units of the TimeSeries Data.

        Returns
        -------
        str:
            Unit sting

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.data.units
        return self._units
