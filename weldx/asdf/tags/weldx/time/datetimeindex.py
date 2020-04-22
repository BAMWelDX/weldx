# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from asdf.yamlutil import custom_tree_to_tagged_tree, tagged_tree_to_custom_tree

from weldx.asdf.types import WeldxType

__all__ = ["DatetimeIndexType"]


class DatetimeIndexType(WeldxType):
    """
    A simple implementation of serializing a single pandas Timedelta.
    """

    name = "time/datetimeindex"
    version = "1.0.0"
    types = [pd.DatetimeIndex]

    @classmethod
    def to_tree(cls, node: pd.DatetimeIndex, ctx):
        """Serialize timedelta to tree."""
        tree = {}
        tree["values"] = custom_tree_to_tagged_tree(node.values.astype(np.int64), ctx)
        tree["start"] = custom_tree_to_tagged_tree(node[0], ctx)
        tree["end"] = custom_tree_to_tagged_tree(node[-1], ctx)
        tree["min"] = custom_tree_to_tagged_tree(node.min(), ctx)
        tree["max"] = custom_tree_to_tagged_tree(node.max(), ctx)
        if node.inferred_freq is not None:
            tree["freq"] = node.inferred_freq
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct timedelta from tree."""
        if "freq" in tree:
            return pd.date_range(
                start=tree["start"], end=tree["end"], freq=tree["freq"]
            )
        values = tagged_tree_to_custom_tree(tree["values"], ctx)
        return pd.DatetimeIndex(values)
