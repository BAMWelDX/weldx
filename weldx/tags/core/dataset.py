from xarray import Dataset

import weldx.tags.core.common_types as ct
from weldx.asdf.types import WeldxConverter


class XarrayDatasetConverter(WeldxConverter):
    """Serialization class for xarray.Dataset"""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/dataset-0.1.*"]
    types = [Dataset]

    def to_yaml_tree(self, obj: Dataset, tag: str, ctx) -> dict:
        """Convert to python dict."""
        attributes = obj.attrs
        coordinates = [
            ct.Variable(name, da.dims, da.data, da.attrs)
            for name, da in obj.coords.items()
        ]
        dimensions = [ct.Dimension(name, length) for name, length in obj.dims.items()]
        variables = [
            ct.Variable(name, da.dims, da.data, da.attrs)
            for name, da in obj.data_vars.items()
        ]

        tree = {
            "attributes": attributes,
            "coordinates": coordinates,
            "dimensions": dimensions,
            "variables": variables,
        }

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        data_vars = {v.name: (v.dimensions, v.data, v.attrs) for v in node["variables"]}
        coords = {c.name: (c.dimensions, c.data, c.attrs) for c in node["coordinates"]}

        return Dataset(data_vars=data_vars, coords=coords, attrs=node["attributes"])
