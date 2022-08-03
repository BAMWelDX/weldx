from weldx.asdf.types import WeldxConverter
from weldx.time import Time

__all__ = ["TimeConverter"]


class TimeConverter(WeldxConverter):
    """A simple implementation of serializing a Time instance."""

    tags = ["asdf://weldx.bam.de/weldx/tags/time/time-0.1.*"]
    types = [Time]

    def to_yaml_tree(self, obj: Time, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = dict()
        tree["values"] = obj.as_pandas()
        tree["reference_time"] = obj._time_ref
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct Time from tree."""
        return Time(node["values"], node.get("reference_time"))

    @staticmethod
    def shape_from_tagged(node):
        """Calculate the shape from static tagged tree instance."""
        from weldx.asdf.util import _get_instance_shape

        return _get_instance_shape(node["values"])
