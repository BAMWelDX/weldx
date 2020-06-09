import numpy as np

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
    def __init__(self, name, shape, data):
        self.name = name
        self.shape = shape
        self.data = data


class NetCDFVariableTypeASDF(WeldxType):
    """Serialization class for a NetCDFVariable"""

    name = "core/netcdf/variable"
    version = "1.0.0"
    types = [NetCDFVariable]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: NetCDFVariable, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        tree = {"name": node.name, "shape": node.shape, "data": node.data}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        return NetCDFVariable(tree["name"], tree["shape"], np.array(tree["data"]))
