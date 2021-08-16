import pandas as pd

from weldx.asdf.types import WeldxConverter

__all__ = ["TimestampConverter"]


class TimestampConverter(WeldxConverter):
    """A simple implementation of serializing a single pandas Timestamp."""

    name = "time/timestamp"
    version = "1.0.0"
    types = [pd.Timestamp]

    def to_yaml_tree(self, obj: pd.Timestamp, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}
        tree["value"] = obj.isoformat()
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct timestamp from tree."""
        return pd.Timestamp(node["value"])
