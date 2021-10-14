import pandas as pd

from weldx.asdf.types import WeldxConverter

__all__ = ["TimestampConverter"]


class TimestampConverter(WeldxConverter):
    """A simple implementation of serializing a single pandas Timestamp."""

    tags = ["asdf://weldx.bam.de/weldx/tags/time/timestamp-0.1.*"]
    types = [pd.Timestamp]

    def to_yaml_tree(self, obj: pd.Timestamp, tag: str, ctx) -> str:
        """Convert to iso format string."""
        return obj.isoformat()

    def from_yaml_tree(self, node: str, tag: str, ctx) -> pd.Timestamp:
        """Construct timestamp from node."""
        # using pd.Timestamp.fromisoformat here would probably be 'better' but
        # there is a bug/regression in some pandas versions that doesn't play nice
        # with nanosecond precision (fails in 1.3.0, works ion 1.3.3)
        return pd.Timestamp(node)
