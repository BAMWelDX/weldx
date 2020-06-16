from xarray import Dataset

import weldx.asdf.tags.weldx.core.common_types as ct
from weldx.asdf.types import WeldxType


class XarrayDatasetASDF(WeldxType):
    """Serialization class for xarray.Dataset"""

    name = "core/dataset"
    version = "1.0.0"
    types = [Dataset]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Dataset, ctx):
        """
        Convert an instance of the 'xarray.Dataset' type into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'xarray.Dataset' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'xarray.Dataset' type to be
            serialized.

        """
        attributes = node.attrs
        coordinates = []
        dimensions = []
        variables = []

        for name, length in dict(node.dims).items():
            dimensions.append(ct.Dimension(name, length))

        for name, data in node.data_vars.items():
            variables.append(ct.Variable(name, data.dims, data.data))

        for name, data in node.coords.items():
            coordinates.append(ct.Variable(name, data.dims, data.data))

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
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
        """
        Converts basic types representing YAML trees into an 'xarray.Dataset'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        xarray.Dataset :
            An instance of the 'xarray.Dataset' type.

        """
        data_vars = {}

        for variable in tree["variables"]:
            data_vars[variable.name] = (variable.dimensions, variable.data)

        coords = {}
        for coordinate in tree["coordinates"]:
            coords[coordinate.name] = (coordinate.dimensions, coordinate.data)

        obj = Dataset(data_vars=data_vars, coords=coords)

        obj.attrs = tree["attributes"]

        return obj
