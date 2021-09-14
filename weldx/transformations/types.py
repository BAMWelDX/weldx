"""shared type definitions."""
from typing import List, Union

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

types_coordinates = Union[xr.DataArray, np.ndarray, List]
types_orientation = Union[xr.DataArray, np.ndarray, List[List], Rotation]


__all__ = [
    "types_coordinates",
    "types_orientation",
]
