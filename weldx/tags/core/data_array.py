"""Serialization for xarray.DataArray."""
from xarray import DataArray

import weldx.tags.core.common_types as ct
from weldx.asdf.types import WeldxConverter


class XarrayDataArrayConverter(WeldxConverter):
    """Serialization class for xarray.DataArray."""

    name = "core/data_array"
    version = "0.1.0"
    types = [DataArray]

    def to_yaml_tree(self, obj: DataArray, tag: str, ctx) -> dict:
        """Convert to python dict."""
        attributes = obj.attrs
        coordinates = [
            ct.Variable(name, coord_data.dims, coord_data.data, attrs=coord_data.attrs)
            for name, coord_data in obj.coords.items()
        ]
        data = ct.Variable("data", obj.dims, obj.data, attrs={})

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "data": data,
        }

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Convert basic types representing YAML trees into an `xarray.DataArray`."""
        data = node["data"].data
        dims = node["data"].dimensions
        coords = {c.name: (c.dimensions, c.data, c.attrs) for c in node["coordinates"]}
        attrs = node["attributes"]

        da = DataArray(data=data, coords=coords, dims=dims, attrs=attrs)
        da.name = None  # we currently do not use the name attribute
        # (but since it gets automatically derived if not set, we define it now.

        return da
