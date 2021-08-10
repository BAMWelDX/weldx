# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import pint

from weldx.asdf.types import WeldxConverter
from weldx.constants import Q_


class PintQuantityConverter(WeldxConverter):
    """A simple implementation of serializing a pint quantity as asdf quantity."""

    tags = ["tag:stsci.edu:asdf/unit/quantity-1.*"]
    types = ["pint.quantity.build_quantity_class.<locals>.Quantity"]

    @classmethod
    def to_yaml_tree(self, obj: pint.Quantity, tag: str, ctx):
        tree = {}
        value = obj.magnitude
        if not value.shape:
            value = value.item()  # convert scalars to native Python numeric types
        tree["value"] = value
        tree["unit"] = str(obj.units)
        return tree

    @classmethod
    def from_yaml_tree(self, node: dict, tag: str, ctx):
        quantity = Q_(node["value"], node["unit"])
        return quantity
