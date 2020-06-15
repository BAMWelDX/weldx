import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from xarray import DataArray

import weldx.asdf.tags.weldx.core.common_types as ct
from weldx.asdf.types import WeldxType


class XarrayDataArrayASDF(WeldxType):
    """Serialization class for xarray.DataArray"""

    name = "core/data_array"
    version = "1.0.0"
    types = [DataArray]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: DataArray, ctx):
        """Convert an xarray.DataArray to a tagged tree"""

        attributes = node.attrs
        coordinates = []
        data = ct.Variable("data", node.dims, node.data)

        for name, coord_data in node.coords.items():
            coordinates.append(ct.Variable(name, coord_data.dims, coord_data.data))

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "data": data,
        }

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.DataArray"""
        data = tree["data"].data
        dims = tree["data"].dimensions
        coords = {}
        for coordinate in tree["coordinates"]:
            coords[coordinate.name] = (coordinate.dimensions, coordinate.data)

        obj = DataArray(data=data, coords=coords, dims=dims)

        obj.attrs = tree["attributes"]

        return obj
