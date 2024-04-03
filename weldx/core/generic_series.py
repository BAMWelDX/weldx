"""Contains GenericSeries class."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pint
import sympy
import xarray as xr
from bidict import bidict
from xarray.core.coordinates import DataArrayCoordinates

from weldx import Q_, U_
from weldx import util as ut
from weldx.constants import UNITS_KEY

from .math_expression import MathematicalExpression

__all__ = ["GenericSeries"]


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

    When calling `weldx.GenericSeries.evaluate`, the passed keyword arguments are
    checked against the dictionaries keys. If a match is found, the corresponding
    preprocessor function is called with the variables value and returns the
    updated value. As an example, this can be used to support multiple time formats.
    The key might be ``t`` and the preprocessor function would turn the original
    time data into an equivalent `xarray.DataArray`.
    """

    _required_dimensions: list[str] = []
    """A list of required dimension names.

    Explicit `weldx.GenericSeries` need all of the listed dimensions.
    Otherwise an exception is raised. If the series is based on an expression,
    the dimension can either be represented by a variable or be part of one
    of the expressions parameters.
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
        obj: pint.Quantity | xr.DataArray | str | MathematicalExpression,
        dims: list[str] | dict[str, str] = None,
        coords: dict[str, list | pint.Quantity] = None,
        units: dict[str, str | pint.Unit] = None,
        interpolation: str = None,
        parameters: dict[str, str | pint.Quantity | xr.DataArray] = None,
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
            use ``dict(t="time")`` to tell the `weldx.GenericSeries` that
            the symbol ``t`` refers to the dimension ``time``.
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
            the provided parameters and dimension units

        Examples
        --------
        Create a `weldx.GenericSeries` representing a translation with 3 m/s
        in x-direction starting at point ``[0, 2 ,2] cm``

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


        The same `weldx.GenericSeries` from above but assigning the ``t`` parameter
        to the output dimension ``time``.

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

        A `weldx.GenericSeries` describing linear interpolation between the values 10 V
        and 20 V over a period of 5 seconds.

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

        self._obj: xr.DataArray | MathematicalExpression = None
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
                obj,
                dims,
                parameters,
                {k: U_(v) for k, v in units.items()},  # catch str
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

    def __hash__(self):
        """Implement as not hashable."""
        return None

    def _init_discrete(
        self,
        data: pint.Quantity | xr.DataArray,
        dims: list[str],
        coords: dict[str, list | pint.Quantity],
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
        expr: str | MathematicalExpression | sympy.Expr,
        dims: dict[str, str],
        parameters: dict[str, str | pint.Quantity | xr.DataArray],
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
            ) from e
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
            ) from e

        return expr_units

    @staticmethod
    def _format_expression_params(
        parameters: dict[str, pint.Quantity | xr.DataArray],
    ) -> dict[str, pint.Quantity | xr.DataArray]:
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

        If the `weldx.GenericSeries` is composed of discrete values, the data
        is interpolated using the specified interpolation method.

        Expressions are simply evaluated if coordinates for all dimensions are provided
        which results in a new discrete `weldx.GenericSeries`. In case that some
        dimensions are left without coordinates, a new expression based
        `weldx.GenericSeries` is returned. The provided coordinates are stored as
        parameters and the corresponding dimensions are no longer variables of
        the new `weldx.GenericSeries`.

        Parameters
        ----------
        kwargs:
            An arbitrary number of keyword arguments. The key must be a dimension name
            of the `weldx.GenericSeries` and the values are the corresponding
            coordinates where the `weldx.GenericSeries` should be evaluated. It is not
            necessary to provide values for all dimensions.
            Partial evaluation is also possible.

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
    def coordinates(self) -> DataArrayCoordinates | None:
        """Get the coordinates of the generic series."""
        if self.is_discrete:
            return self.data_array.coords
        return None

    @property
    def coordinate_names(self) -> list[str]:
        """Get the names of all coordinates."""
        return NotImplemented

    @property
    def data(self) -> pint.Quantity | MathematicalExpression:
        """Get the internal data."""
        if self.is_discrete:
            return self.data_array.data
        return self._obj

    @property
    def data_array(self) -> xr.DataArray | None:
        """Get the internal data as `xarray.DataArray`."""
        if self.is_discrete:
            return self._obj
        return None

    @staticmethod
    def _get_expression_dims(
        expr: MathematicalExpression, symbol_dims: Mapping[str, str]
    ) -> list[str]:
        """Get the dimensions of an expression based `weldx.GenericSeries`.

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
        """Set the interpolation."""
        if val not in (
            "linear",
            "step",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
        ):
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
    def variable_names(self) -> list[str] | None:
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

        # check dimensionality constraint and coordinates
        cls._check_units_and_coords(expr, expr_units, var_dims, var_units)

    @classmethod
    def _check_units_and_coords(cls, expr, expr_units, var_dims, var_units):
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
        """Add two `weldx.GenericSeries`."""
        # this should mostly be moved to the MathematicalExpression
        # todo:
        #   - for two expressions simply do: f"{exp_1} + f{exp_2}" and merge the
        #     parameters in a new MathExpression
        #   - for two discrete series call __add__ of the xarrays
        #   - for mixed version add a new parameter to the expression string and set the
        #     xarray as the parameters value
        return NotImplemented

    @staticmethod
    def interp_like(
        obj: Any,  # skipcq: PYL-W0613
        dimensions: list[str] = None,  # skipcq: PYL-W0613
        accessor_mappings: dict = None,  # skipcq: PYL-W0613
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


@dataclass
class SeriesParameter:
    """Describes a parameter/coordinate of a Series and convert between formats.

    The input value gets stored as either quantity or DataArray.
    (DataArray is stored 'as is', other inputs will be converted to quantities).

    In addition, the desired dimension on the Parameter and an optional symbol
    representation for math expressions can be added.

    The stored value can be converted to different formats available as properties.
    """

    values: xr.DataArray | pint.Quantity
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
            return self.values.units  # type: ignore[return-value]
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


def _quantity_to_coord_tuple(
    v: pint.Quantity, dim
) -> tuple[str, np.ndarray, dict[str, pint.Unit]]:
    return dim, v.m, {UNITS_KEY: v.u}  # type: ignore[dict-item]


def _quantity_to_xarray(v: pint.Quantity, dim: str = None) -> xr.DataArray:
    """Convert a single quantity into a formatted xarray dataarray."""
    return xr.DataArray(v, dims=dim)


def _quantities_to_xarray(q_dict: dict[str, pint.Quantity]) -> dict[str, xr.DataArray]:
    """Convert a str:Quantity mapping into a mapping of `xarray.DataArray`."""
    return {k: _quantity_to_xarray(v, k) for k, v in q_dict.items()}
