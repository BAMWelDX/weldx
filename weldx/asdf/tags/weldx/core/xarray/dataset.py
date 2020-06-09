from xarray import Dataset

from weldx.asdf.types import WeldxType
import weldx.asdf.tags.weldx.core.netcdf as netcdf


class XarrayDatasetASDF(WeldxType):
    """Serialization class for xarray.Dataset"""

    name = "core/netcdf/file_format"
    version = "1.0.0"
    types = [Dataset]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Dataset, ctx):
        """Convert an xarray.Dataset to a tagged tree"""
        attributes = []
        dimensions = []
        variables = []

        for name, length in dict(node.dims).items():
            dimensions.append(netcdf.NetCDFDimension(name, length))

        for name, data in node.data_vars.items():
            variables.append(netcdf.NetCDFVariable(name, data.dims, data.data))

        coordinates = []
        for name, data in node.coords.items():
            variables.append(netcdf.NetCDFVariable(name, data.dims, data.data))
            coordinates.append(name)

        attributes.append(netcdf.NetCDFAttribute("coordinates", coordinates))

        tree = {
            "attributes": attributes,
            "dimensions": dimensions,
            "variables": variables,
        }

        # variables = []
        # for variable_name in node:
        #    variables.append(netcdf.NetCDFVariable(node[variable_name]))
        # tree = {"coordinates": dict(node.coords), "variables": variables}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert a tagged tree to an xarray.Dataset"""
        coordinate_names = None
        for attribute in tree["attributes"]:
            if attribute.name == "coordinates":
                coordinate_names = attribute.data

        data_vars = {}
        coords = {}
        for variable in tree["variables"]:
            if variable.name in coordinate_names:
                coords[variable.name] = (variable.shape, variable.data)
            else:
                data_vars[variable.name] = (variable.shape, variable.data)

        return Dataset(data_vars=data_vars, coords=coords)
