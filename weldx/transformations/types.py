"""shared type definitions."""
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:  # pragma: no cover
    import weldx

types_coordinates = Union[xr.DataArray, np.ndarray, List]
types_orientation = Union[xr.DataArray, np.ndarray, List[List], Rotation]
types_timeindex = Union[pd.DatetimeIndex, pd.TimedeltaIndex, pint.Quantity]
types_time_and_lcs = Union[
    types_timeindex, "weldx.transformations.LocalCoordinateSystem"
]


@runtime_checkable
class SupportsTime(Protocol):
    """An ABC with one abstract method time."""

    __slots__ = ()

    @abstractmethod
    def time(self):  # noqa
        raise NotImplementedError


__all__ = [
    "SupportsTime",
    "types_coordinates",
    "types_orientation",
    "types_timeindex",
    "types_time_and_lcs",
]
