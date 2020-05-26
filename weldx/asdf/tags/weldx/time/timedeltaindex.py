# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from asdf.yamlutil import custom_tree_to_tagged_tree, tagged_tree_to_custom_tree

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
            tree["freq"] = custom_tree_to_tagged_tree(node.inferred_freq, ctx)
        else:
            tree["values"] = custom_tree_to_tagged_tree(
                node.values.astype(np.int64), ctx
            )

        tree["start"] = custom_tree_to_tagged_tree(node[0], ctx)
        tree["end"] = custom_tree_to_tagged_tree(node[-1], ctx)
        tree["min"] = custom_tree_to_tagged_tree(node.min(), ctx)
        tree["max"] = custom_tree_to_tagged_tree(node.max(), ctx)

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct TimedeltaIndex from tree."""
        if "freq" in tree:
            return pd.timedelta_range(
                start=tree["start"], end=tree["end"], freq=tree["freq"]
            )
        values = tagged_tree_to_custom_tree(tree["values"], ctx)
        return pd.TimedeltaIndex(values)
