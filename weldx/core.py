"""Collection of common classes and functions."""

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pint
import sympy
import xarray as xr

import weldx.utility as ut
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG


class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(
        self, expression: sympy.core.basic.Basic, parameters: Union[Dict, None] = None
    ):
        """Initialize a MathematicalExpression from sympy objects.

        Parameters
        ----------
        expression
            sympy object that can be turned into an expression.
            E.g. any valid combination of previously defined sympy.symbols.
        parameters :
            A dictionary containing constant values for variables of the
            expression.

        """
        if isinstance(expression, str):
            expression = sympy.sympify(expression)
        self.expression = expression
        self.function = sympy.lambdify(
            self.expression.free_symbols, self.expression, "numpy"
        )
        self.parameters = {}
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError('"parameters" must be dictionary')
            variable_names = self.get_variable_names()
            for key in parameters:
                if key not in variable_names:
                    raise ValueError(
                        f'The expression does not have a parameter "{key}"'
                    )
            self.parameters = parameters

    def set_parameter(self, name, value):
        """Define an expression parameter as constant value.

        Parameters
        ----------
        name
            Name of the parameter used in the expression.
        value
            Parameter value. This can be number, array or weldx.Quantity

        """
        self.parameters[name] = value

    @property
    def num_parameters(self):
        """Get the expressions number of parameters.

        Returns
        -------
        int:
            Number of parameters.

        """
        return len(self.parameters)

    @property
    def num_variables(self):
        """Get the expressions number of free variables.

        Returns
        -------
        int:
            Number of free variables.

        """
        return len(self.expression.free_symbols) - len(self.parameters)

    def get_variable_names(self):
        """Get a list of all expression variables.

        Returns
        -------
        List

        """
        variable_names = []
        for var in self.expression.free_symbols:
            if var.__str__() not in self.parameters:
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
        intersection = set(kwargs).intersection(self.parameters)
        if len(intersection) > 0:
            raise ValueError(
                f"The variables {intersection} are already defined as parameters."
            )
        inputs = {**kwargs, **self.parameters}
        return self.function(**inputs)


# TimeSeries ---------------------------------------------------------------------------


class TimeSeries:
    """Describes the behaviour of a quantity in time."""

    _valid_interpolations = ["step", "linear"]

    def __init__(
        self,
        data: Union[Q_, MathematicalExpression],
        time: Union[None, pd.TimedeltaIndex] = None,
        interpolation: str = "linear",
    ):
        """Construct a TimSeries.

        Parameters
        ----------
        data:
            Either a weldx Quantity or a weldx.MathematicalExpression. If a mathematical
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

        if isinstance(data, Q_):
            if not isinstance(data.magnitude, np.ndarray):
                data = Q_([data.magnitude], data.units)
                if time is None:
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

    def interp_time(self, time: Union[pd.TimedeltaIndex, Q_]):
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

        if not isinstance(time, Q_) or not time.check(UREG.get_dimensionality("s")):
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
