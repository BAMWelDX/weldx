# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pint
from asdf.yamlutil import custom_tree_to_tagged_tree, tagged_tree_to_custom_tree

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
        tree["value"] = custom_tree_to_tagged_tree(node.magnitude, ctx)
        tree["unit"] = custom_tree_to_tagged_tree(str(node.units), ctx)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        value = tagged_tree_to_custom_tree(tree["value"], ctx)
        unit = tagged_tree_to_custom_tree(tree["unit"], ctx)
        quantity = Q_(value, unit)
        return quantity
