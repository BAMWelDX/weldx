# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pint

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.types import WeldxAsdfType

__all__ = ["PintQuantityType"]


class PintQuantityType(WeldxAsdfType):
    """A simple implementation of serializing a pint quantity as asdf quantity."""

    name = "unit/quantity"
    version = "1.1.0"
    types = [pint.Quantity]

    @classmethod
    def to_tree(cls, node: pint.Quantity, ctx):
        tree = {}
        tree["value"] = node.magnitude
        tree["unit"] = str(node.units)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        value = tree["value"]
        unit = tree["unit"]
        quantity = Q_(value, unit)
        return quantity
