from typing import List

import numpy as np
import pandas as pd
from asdf.tagged import TaggedDict

from weldx.asdf.types import WeldxConverter

__all__ = ["TimedeltaIndexConverter"]


class TimedeltaIndexConverter(WeldxConverter):
    """A simple implementation of serializing pandas TimedeltaIndex."""

    name = "time/timedeltaindex"
    version = "0.1.*"
    types = [pd.TimedeltaIndex]

    def to_yaml_tree(self, obj: pd.TimedeltaIndex, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}
        if obj.inferred_freq is not None:
            tree["freq"] = obj.inferred_freq
        else:
            tree["values"] = obj.values.astype(np.int64)

        tree["start"] = obj[0]
        tree["end"] = obj[-1]
        tree["min"] = obj.min()
        tree["max"] = obj.max()

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct TimedeltaIndex from tree."""
        if "freq" in node:
            return pd.timedelta_range(
                start=node["start"], end=node["end"], freq=node["freq"]
            )
        values = node["values"]
        return pd.TimedeltaIndex(values)

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> List[int]:
        """Calculate the shape from static tagged tree instance."""
        _tag: str = node._tag

        if "freq" in node:
            if _tag.startswith("tag:weldx.bam.de:weldx"):  # legacy_code
                tdi_temp = pd.timedelta_range(
                    start=node["start"]["value"],
                    end=node["end"]["value"],
                    freq=node["freq"],
                )
                return [len(tdi_temp)]
            tdi_temp = pd.timedelta_range(
                start=str(node["start"]),  # can't handle TaggedString directly
                end=str(node["end"]),
                freq=node["freq"],
            )
            return [len(tdi_temp)]
        return node["values"]["shape"]
