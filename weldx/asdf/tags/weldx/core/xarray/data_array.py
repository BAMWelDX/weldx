import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from xarray import DataArray

import weldx.asdf.tags.weldx.core.netcdf as netcdf
from weldx.asdf.types import WeldxType


class XarrayDataArrayASDF(WeldxType):
    """Serialization class for xarray.DataArray"""

    name = "core/xarray/data_array"
    version = "1.0.0"
    types = [DataArray]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: DataArray, ctx):
        """Convert an xarray.DataArray to a tagged tree"""

        attributes = []
        coordinates = []
        variables = netcdf.NetCDFVariable("data", node.dims, node.data)

        for name, data in node.coords.items():
            coordinates.append(netcdf.NetCDFVariable(name, data.dims, data.data))

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "variables": variables,
        }

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.DataArray"""
        data = tree["variables"].data
        dims = tree["variables"].dimensions
        coords = {}
        for coordinate in tree["coordinates"]:
            coords[coordinate.name] = (coordinate.dimensions, coordinate.data)
        return DataArray(data=data, coords=coords, dims=dims)
