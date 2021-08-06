"""shared type definitions."""
from abc import abstractmethod
from typing import List, Protocol, Union, runtime_checkable

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

types_coordinates = Union[xr.DataArray, np.ndarray, List]
types_orientation = Union[xr.DataArray, np.ndarray, List[List], Rotation]


# remove ???
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
]
