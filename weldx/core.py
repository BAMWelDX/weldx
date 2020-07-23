"""Collection of common classes and functions."""

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pint
import sympy
import xarray as xr
from asdf.tags.core.ndarray import NDArrayType

import weldx.utility as ut
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG


# TODO: Add __repr__ functions
class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(
        self, expression: Union[sympy.Expr, str], parameters: Union[Dict, None] = None
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
        self._expression = expression
        self.function = sympy.lambdify(
            self._expression.free_symbols, self._expression, "numpy"
        )
        self._parameters = {}
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError('"parameters" must be dictionary')
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
            representation += f"\tNone"
        return representation

    def __eq__(self, other):
        """Return the result of a structural equality comparison with another object.

        If the other object is not a MathematicalExpression this function always returns
        'False'.

        Parameters
        ----------
        other:
            Other object.

        Returns
        -------
        bool:
            'True' if the compared object is also a Mathematical expression and equal to
             this instance, 'False' otherwise

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
                equality = sympy.simplify(self.expression - other.expression) == 0

            if check_parameters:
                equality = equality and self._parameters == other.parameters
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
        """
        Return the internal sympy expression.

        Returns
        -------
        Internal sympy expression

        """
        return self._expression

    @property
    def parameters(self) -> Dict:
        """
        Return the internal parameters dictionary.

        Returns
        -------
        Dict
            Internal parameters dictionary

        """
        return self._parameters

    def get_variable_names(self):
        """Get a list of all expression variables.

        Returns
        -------
        List

        """
        variable_names = []
        for var in self._expression.free_symbols:
            if var.__str__() not in self._parameters:
                variable_names.append(var.__str__())
        return variable_names

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

    _valid_interpolations = ["step", "linear"]

    def __init__(
        self,
        data: Union[pint.Quantity, MathematicalExpression],
        time: Union[None, pd.TimedeltaIndex] = None,
        interpolation: str = "linear",
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
        self._time = None
        self._data = None
        self._interpolation = None
        self._time_var_name = None
        self._shape = None
        self._units = None

        if isinstance(data, pint.Quantity):
            if not np.iterable(data.magnitude):
                data = Q_([data.magnitude], data.units)
            if time is None and data.shape[0] == 1:
                time = pd.TimedeltaIndex([0])

            if interpolation not in self._valid_interpolations:
                raise ValueError(
                    "A valid interpolation method must be specified "
                    f'if discrete values are used. "{interpolation}" is not supported'
                )

            self._data = xr.DataArray(data=data, dims=["time"], coords={"time": time})
            self._interpolation = interpolation

        elif isinstance(data, MathematicalExpression):

            if data.num_variables != 1:
                raise Exception(
                    "The mathematical expression must have exactly 1 free "
                    "variable that represents time."
                )
            time_var_name = data.get_variable_names()[0]
            try:
                eval_data = data.evaluate(**{time_var_name: Q_(1, "second")})
                self._units = eval_data.units
                if isinstance(eval_data.magnitude, np.ndarray):
                    self._shape = eval_data.magnitude.shape
                else:
                    self._shape = tuple([1])
            except pint.errors.DimensionalityError:
                raise Exception(
                    "Expression can not be evaluated with "
                    '"weldx.Quantity(1, "seconds")"'
                    ". Ensure that every parameter posses the correct unit."
                )

            self._data = data
            self._time_var_name = time_var_name

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

        else:
            raise TypeError(f'The data type "{type(data)}" is not supported.')

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return False
        if isinstance(self.data, pint.Quantity):
            if not isinstance(other.data, pint.Quantity):
                return False
            return (
                np.all(self.data == other.data)
                and np.all(self.time == other.time)
                and self._interpolation == other.interpolation
            )

        return self._data == other.data

    def __repr__(self):
        """Give __repr__ output."""
        representation = f"<TimeSeries>"
        if isinstance(self._data, xr.DataArray):
            if self._data.shape[0] == 1:
                representation += f"\nConstant value:\n\t{self.data.magnitude[0]}\n"
            else:
                representation += (
                    f"\nTime:\n\t{self.time}\n"
                    + f"Values:\n\t{self.data.magnitude}\n"
                    + f"Interpolation:\n\t{self.interpolation}\n"
                )
        else:
            representation += self.data.__repr__().replace(
                "<MathematicalExpression>", ""
            )
        return representation + f"Units:\n\t{self.units}\n"

    @property
    def data(self) -> Union[xr.DataArray, MathematicalExpression]:
        """Return the data of the TimeSeries.

        This is either a set of discrete values or a mathematical expression.

        Returns
        -------
        xr.DataArray:
            Underlying data array with discrete values in time.
        MathematicalExpression:
            A mathematical expression describing the time dependency

        """
        if isinstance(self._data, xr.DataArray):
            return self._data.data
        return self._data

    @property
    def interpolation(self) -> Union[str, None]:
        """Return the interpolation.

        Returns
        -------
        str:
            Interpolation of the TimeSeries

        """
        return self._interpolation

    @property
    def time(self) -> Union[None, pd.TimedeltaIndex]:
        """Return the data's timestamps.

        Returns
        -------
        pd.TimedeltaIndex:
            Timestamps of the  data

        """
        if isinstance(self._data, xr.DataArray):
            if len(self._data.time) == 1:
                return None
            return ut.to_pandas_time_index(self._data.time.data)
        return self._time

    def interp_time(self, time: Union[pd.TimedeltaIndex, pint.Quantity]):
        """Interpolate the TimeSeries in time.

        If the internal data consists of discrete values, an interpolation with the
        prescribed interpolation method is performed. In case of mathematical
        expression, the expression is evaluated for the given timestamps.

        Parameters
        ----------
        time:
            A set of timestamps.

        Returns
        -------
        xr.DataArray:
            A data array containing the interpolated data.

        """
        if isinstance(self._data, xr.DataArray):
            if self._interpolation == "linear":
                return ut.xr_interp_like(
                    self._data,
                    {"time": time},
                    assume_sorted=False,
                    broadcast_missing=False,
                )
            raise Exception("not implemented")

        if not isinstance(time, pint.Quantity) or not time.check(
            UREG.get_dimensionality("s")
        ):
            raise ValueError('"time" must be a time quantity.')

        if len(self.shape) > 1 and isinstance(time.magnitude, np.ndarray):
            while len(time.magnitude.shape) < len(self.shape):
                time = Q_(time.magnitude[:, np.newaxis], time.units)

        # evaluate expression
        data = self._data.evaluate(**{self._time_var_name: time})

        # create data array
        if not np.iterable(data.magnitude):  # make sure quantity is not scalar value
            data = data * np.array([1])
        if hasattr(time, "shape"):  # squeeze out any helper dimensions
            time = np.squeeze(time)
        time = ut.to_pandas_time_index(time)
        dax = xr.DataArray(data=data)  # don't know exact dimensions so far
        dax = dax.rename({"dim_0": "time"}).assign_coords({"time": time})
        return dax

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
