# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pandas as pd

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
        tree["value"] = node.isoformat()
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct timestamp from tree."""
        value = tree["value"]
        return pd.Timestamp(value)
