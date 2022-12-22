from copy import copy

from weldx.asdf.types import WeldxConverter
from weldx.geometry import SpatialData

__all__ = ["SpatialDataConverter"]


class SpatialDataConverter(WeldxConverter):
    """Converter for SpatialData."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/geometry/spatial_data-0.1.*"]
    types = [SpatialData]

    def to_yaml_tree(self, obj: SpatialData, tag: str, ctx) -> dict:
        """Serialize into tree."""
        tree = copy(obj.__dict__)  # shallow copy so we dont change original object
        if tree["coordinates"].ndim <= 2:
            tree["coordinates"] = tree["coordinates"].data
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx) -> SpatialData:
        """Reconstruct from yaml node."""
        from weldx.constants import Q_

        if tag == "asdf://weldx.bam.de/weldx/tags/core/geometry/spatial_data-0.1.0":
            node["coordinates"] = Q_(node["coordinates"], "mm")  # legacy
        return SpatialData(**node)
