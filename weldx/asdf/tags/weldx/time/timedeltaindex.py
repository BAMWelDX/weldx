# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pandas as pd
from asdf.tagged import TaggedDict

from weldx.asdf.types import WeldxType

__all__ = ["TimedeltaIndexType"]


class TimedeltaIndexType(WeldxType):
    """A simple implementation of serializing pandas TimedeltaIndex."""

    name = "time/timedeltaindex"
    version = "1.0.0"
    types = [pd.TimedeltaIndex]

    @classmethod
    def to_tree(cls, node: pd.TimedeltaIndex, ctx):
        """Serialize TimedeltaIndex to tree."""
        tree = {}
        if node.inferred_freq is not None:
            tree["freq"] = node.inferred_freq
        else:
            tree["values"] = node.values.astype(np.int64)

        tree["start"] = node[0]
        tree["end"] = node[-1]
        tree["min"] = node.min()
        tree["max"] = node.max()

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct TimedeltaIndex from tree."""
        if "freq" in tree:
            return pd.timedelta_range(
                start=tree["start"], end=tree["end"], freq=tree["freq"]
            )
        values = tree["values"]
        return pd.TimedeltaIndex(values)

    @staticmethod
    def shape_from_tagged(tree: TaggedDict) -> List[int]:
        """Calculate the shape (length of TDI) from static tagged tree instance."""
        if "freq" in tree:
            tdi_temp = pd.timedelta_range(
                start=tree["start"]["value"],
                end=tree["end"]["value"],
                freq=tree["freq"],
            )
            return [len(tdi_temp)]
        else:
            return tree["values"]["shape"]
