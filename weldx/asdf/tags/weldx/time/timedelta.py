import pandas as pd
from asdf.extension import Converter

from weldx.asdf.types import WeldxConverterMeta

__all__ = ["TimedeltaConverter"]


class TimedeltaConverter(Converter, metaclass=WeldxConverterMeta):
    """A simple implementation of serializing a single pandas Timedelta."""

    tags = ["asdf://weldx.bam.de/weldx/tags/time/timedelta-1.*"]
    types = [pd.Timedelta]

    @classmethod
    def to_yaml_tree(self, obj: pd.Timedelta, tag, ctx):
        """Serialize timedelta to tree."""
        tree = {}
        tree["value"] = obj.isoformat()
        return tree

    @classmethod
    def from_yaml_tree(self, node, tag, ctx):
        """Construct timedelta from tree."""
        value = node["value"]
        return pd.Timedelta(value)
