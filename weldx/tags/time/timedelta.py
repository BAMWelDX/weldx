import pandas as pd

from weldx.asdf.types import WeldxConverter

__all__ = ["TimedeltaConverter"]


class TimedeltaConverter(WeldxConverter):
    """A simple implementation of serializing a single pandas Timedelta."""

    tags = ["asdf://weldx.bam.de/weldx/tags/time/timedelta-0.1.*"]
    types = [pd.Timedelta]

    def to_yaml_tree(self, obj: pd.Timedelta, tag, ctx) -> str:
        """Convert to iso format string."""
        return obj.isoformat()

    def from_yaml_tree(self, node: str, tag, ctx) -> pd.Timedelta:
        """Construct timedelta from tree."""
        return pd.Timedelta(node)
