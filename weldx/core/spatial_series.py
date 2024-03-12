"""Contains the SpatialSeries class."""

from __future__ import annotations

import pint
import xarray as xr

from weldx import Q_, U_

from .generic_series import GenericSeries
from .math_expression import MathematicalExpression

__all__ = ["SpatialSeries"]


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
        obj: pint.Quantity | xr.DataArray | str | MathematicalExpression,
        dims: list[str] | dict[str, str] = None,
        coords: dict[str, list | pint.Quantity] = None,
        units: dict[str, str | pint.Unit] = None,
        interpolation: str = None,
        parameters: dict[str, str | pint.Quantity | xr.DataArray] = None,
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
        obj: pint.Quantity | xr.DataArray | str | MathematicalExpression,
        dims: list[str] | dict[str, str],
        coords: dict[str, list | pint.Quantity],
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
