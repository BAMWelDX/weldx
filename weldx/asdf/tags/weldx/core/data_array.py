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
        """
        Convert an instance of the 'xarray.DataArray' type into YAML  representations.

        Parameters
        ----------
        node :
            Instance of the 'xarray.DataArray' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'xarray.DataArray'
            type to be serialized.

        """
        attributes = node.attrs
        coordinates = [
            ct.Variable(name, coord_data.dims, coord_data.data, attrs=coord_data.attrs)
            for name, coord_data in node.coords.items()
        ]
        data = ct.Variable("data", node.dims, node.data, attrs={})

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "data": data,
        }

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into an 'xarray.DataArray'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        xarray.DataArray :
            An instance of the 'xarray.DataArray' type.

        """
        data = tree["data"].data
        dims = tree["data"].dimensions
        coords = {c.name: (c.dimensions, c.data, c.attrs) for c in tree["coordinates"]}
        attrs = tree["attributes"]

        da = DataArray(data=data, coords=coords, dims=dims, attrs=attrs)
        return da
