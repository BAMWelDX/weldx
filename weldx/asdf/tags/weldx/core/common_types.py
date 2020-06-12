import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from weldx.asdf.types import WeldxType


class Attribute:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class AttributeTypeASDF(WeldxType):
    """Serialization class for a Attribute"""

    name = "core/attribute"
    version = "1.0.0"
    types = [Attribute]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Attribute, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        tree = {"name": node.name, "data": node.data}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        return Attribute(tree["name"], tree["data"])


class Dimension:
    def __init__(self, name, length):
        self.name = name
        self.length = length


class DimensionTypeASDF(WeldxType):
    """Serialization class for a Dimension"""

    name = "core/dimension"
    version = "1.0.0"
    types = [Dimension]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Dimension, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        tree = {"name": node.name, "length": node.length}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        return Dimension(tree["name"], tree["length"])


class Variable:
    def __init__(self, name, dimensions, data: np.ndarray):
        self.name = name
        self.dimensions = dimensions
        self.data = data


class VariableTypeASDF(WeldxType):
    """Serialization class for a Variable"""

    name = "core/variable"
    version = "1.0.0"
    types = [Variable]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @staticmethod
    def convert_time_dtypes(data: np.ndarray):
        if is_datetime(data.dtype):
            return data.astype(np.int64)
        return data

    @classmethod
    def to_tree(cls, node: Variable, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        dtype = node.data.dtype.str
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
        return Variable(tree["name"], tree["dimensions"], data)
