from __future__ import annotations

import numpy as np
import pandas as pd
from asdf.tagged import TaggedDict

from weldx.asdf.types import WeldxConverter

__all__ = ["TimedeltaIndexConverter"]

PANDAS_OLD_UNIT_SUFFIXES = dict(H="h", T="min", S="s", L="ms", U="us", N="ns")


def _handle_converted_pd_tdi_units(node: TaggedDict):
    """Convert changed units in Pandas.Datetimeindex to valid values."""
    for suf in PANDAS_OLD_UNIT_SUFFIXES:
        node["freq"] = node["freq"].replace(suf, PANDAS_OLD_UNIT_SUFFIXES[suf])


class TimedeltaIndexConverter(WeldxConverter):
    """A simple implementation of serializing pandas TimedeltaIndex."""

    tags = ["asdf://weldx.bam.de/weldx/tags/time/timedeltaindex-0.1.*"]
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
            _handle_converted_pd_tdi_units(node)
            return pd.timedelta_range(
                start=node["start"], end=node["end"], freq=node["freq"]
            )
        values = node["values"]
        return pd.TimedeltaIndex(values)

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> list[int]:
        """Calculate the shape from static tagged tree instance."""
        if "freq" in node:
            _handle_converted_pd_tdi_units(node)
            tdi_temp = pd.timedelta_range(
                start=str(node["start"]),  # can't handle TaggedString directly
                end=str(node["end"]),
                freq=node["freq"],
            )
            return [len(tdi_temp)]
        return node["values"]["shape"]
