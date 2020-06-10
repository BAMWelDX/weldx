import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from weldx.asdf.types import WeldxType


class NetCDFAttribute:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class NetCDFAttributeTypeASDF(WeldxType):
    """Serialization class for a NetCDFAttribute"""

    name = "core/netcdf/attribute"
    version = "1.0.0"
    types = [NetCDFAttribute]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: NetCDFAttribute, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        tree = {"name": node.name, "data": node.data}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        return NetCDFAttribute(tree["name"], tree["data"])


class NetCDFDimension:
    def __init__(self, name, length):
        self.name = name
        self.length = length


class NetCDFDimensionTypeASDF(WeldxType):
    """Serialization class for a NetCDFDimension"""

    name = "core/netcdf/dimension"
    version = "1.0.0"
    types = [NetCDFDimension]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: NetCDFDimension, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        tree = {"name": node.name, "length": node.length}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        return NetCDFDimension(tree["name"], tree["length"])


class NetCDFVariable:
    def __init__(self, name, dimensions, data: np.ndarray):
        self.name = name
        self.dimensions = dimensions
        self.data = data


class NetCDFVariableTypeASDF(WeldxType):
    """Serialization class for a NetCDFVariable"""

    name = "core/netcdf/variable"
    version = "1.0.0"
    types = [NetCDFVariable]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @staticmethod
    def convert_time_dtypes(data: np.ndarray):
        if is_datetime(data.dtype):
            return data.astype(np.int64)
        return data

    @classmethod
    def to_tree(cls, node: NetCDFVariable, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        dtype = node.data.dtype.name
        data = cls.convert_time_dtypes(data=node.data)
        tree = {
            "name": node.name,
            "dimensions": node.dimensions,
            "dtype": dtype,
            "data": data,
        }

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        dtype = np.dtype(tree["dtype"])
        data = np.array(tree["data"]).astype(dtype)
        return NetCDFVariable(tree["name"], tree["dimensions"], data)
