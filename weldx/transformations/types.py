"""shared type definitions."""
from typing import Union

import numpy.typing as npt
import pint
import xarray as xr
from scipy.spatial.transform import Rotation

types_coordinates = Union[xr.DataArray, npt.ArrayLike, pint.Quantity]
types_orientation = Union[xr.DataArray, npt.ArrayLike, Rotation]


__all__ = [
    "types_coordinates",
    "types_orientation",
]
