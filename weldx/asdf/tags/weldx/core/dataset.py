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
        coordinates = [
            ct.Variable(name, da.dims, da.data, da.attrs)
            for name, da in node.coords.items()
        ]
        dimensions = [ct.Dimension(name, length) for name, length in node.dims.items()]
        variables = [
            ct.Variable(name, da.dims, da.data, da.attrs)
            for name, da in node.data_vars.items()
        ]

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "dimensions": dimensions,
            "variables": variables,
        }

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
        data_vars = {v.name: (v.dimensions, v.data, v.attrs) for v in tree["variables"]}
        coords = {c.name: (c.dimensions, c.data, c.attrs) for c in tree["coordinates"]}

        return Dataset(data_vars=data_vars, coords=coords, attrs=tree["attributes"])
