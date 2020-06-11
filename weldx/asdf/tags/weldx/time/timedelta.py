# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pandas as pd

from weldx.asdf.types import WeldxType

__all__ = ["TimedeltaType"]


class TimedeltaType(WeldxType):
    """A simple implementation of serializing a single pandas Timedelta."""

    name = "time/timedelta"
    version = "1.0.0"
    types = [pd.Timedelta]

    @classmethod
    def to_tree(cls, node: pd.Timedelta, ctx):
        """Serialize timedelta to tree."""
        tree = {}
        tree["value"] = node.isoformat()
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """Construct timedelta from tree."""
        value = tree["value"]
        return pd.Timedelta(value)
