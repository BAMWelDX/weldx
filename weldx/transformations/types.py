from abc import abstractmethod
from typing import List, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation

types_coordinates = Union[xr.DataArray, np.ndarray, List]
types_orientation = Union[xr.DataArray, np.ndarray, List[List], Rotation]
types_time = Union[pd.DatetimeIndex, pd.TimedeltaIndex, pint.Quantity]
types_time_and_lcs = Union[types_time, "weldx.transformations.LocalCoordinateSystem"]


@runtime_checkable
class SupportsTime(Protocol):
    """An ABC with one abstract method time."""

    __slots__ = ()

    @abstractmethod
    def time(self):
        pass


__all__ = [
    "SupportsTime",
    "types_coordinates",
    "types_orientation",
    "types_time",
    "types_time_and_lcs",
]
