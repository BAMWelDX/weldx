# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pandas as pd
from asdf.yamlutil import custom_tree_to_tagged_tree, tagged_tree_to_custom_tree

from weldx.asdf.types import WeldxType

__all__ = ["TimestampType"]


class TimestampType(WeldxType):
    """A simple implementation of serializing a single pandas Timestamp."""

    name = "time/timestamp"
    version = "1.0.0"
    types = [pd.Timestamp]

    @classmethod
    def to_tree(cls, node: pd.Timestamp, ctx):
        """Serialize timestamp to tree."""
        tree = {}
        tree["value"] = custom_tree_to_tagged_tree(node.isoformat(), ctx)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct timestamp from tree."""
        value = tagged_tree_to_custom_tree(tree["value"], ctx)
        return pd.Timestamp(value)
