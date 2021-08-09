from weldx.asdf.types import WeldxConverter
from weldx.time import Time

__all__ = ["TimeType"]


class TimeType(WeldxConverter):
    """A simple implementation of serializing a Time instance."""

    name = "time/time"
    version = "1.0.0"
    types = [Time]

    @classmethod
    def to_tree(cls, node: Time, ctx):
        """Serialize timedelta to tree."""
        tree = dict()
        tree["values"] = node.as_pandas()
        tree["reference_time"] = node._time_ref
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct Time from tree."""
        return Time(tree["values"], tree.get("reference_time"))
