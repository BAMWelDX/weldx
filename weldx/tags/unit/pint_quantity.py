from typing import List

import pint
from asdf.tagged import TaggedDict

from weldx.asdf.types import WeldxConverter
from weldx.asdf.util import _get_instance_shape
from weldx.constants import Q_


class PintQuantityConverter(WeldxConverter):
    """A simple implementation of serializing a pint quantity as asdf quantity."""

    tags = ["tag:stsci.edu:asdf/unit/quantity-1.*"]
    types = ["pint.quantity.build_quantity_class.<locals>.Quantity"]

    def to_yaml_tree(self, obj: pint.Quantity, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}
        value = obj.magnitude
        if not value.shape:
            value = value.item()  # convert scalars to native Python numeric types
        tree["value"] = value
        tree["unit"] = str(obj.units)
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Reconstruct from tree."""
        quantity = Q_(node["value"], node["unit"])
        return quantity

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> List[int]:
        """Calculate the shape from static tagged tree instance."""
        if isinstance(node["value"], dict):  # ndarray
            return _get_instance_shape(node["value"])
        return [1]  # scalar
