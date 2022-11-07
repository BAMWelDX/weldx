from __future__ import annotations

import pint
from asdf.tagged import TaggedDict

from weldx.asdf.constants import WELDX_TAG_URI_BASE
from weldx.asdf.types import WeldxConverter
from weldx.asdf.util import _get_instance_shape
from weldx.constants import Q_, U_


class PintQuantityConverter(WeldxConverter):
    """A simple implementation of serializing a pint quantity as asdf quantity."""

    tags = [
        "asdf://weldx.bam.de/weldx/tags/units/quantity-0.1.*",
        "tag:stsci.edu:asdf/unit/quantity-1.*",
    ]
    types = [
        "pint.util.Quantity",  # pint >= 0.20
        "pint.quantity.build_quantity_class.<locals>.Quantity",  # pint < 0.20
        "weldx.constants.Q_",
    ]

    def to_yaml_tree(self, obj: pint.Quantity, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}
        value = obj.magnitude
        if not value.shape:
            value = value.item()  # convert scalars to native Python numeric types
        tree["value"] = value
        tree["units"] = obj.units
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Reconstruct from tree."""
        if tag.startswith("tag:stsci.edu:asdf"):  # asdf compat
            return Q_(node["value"], node["unit"])
        return Q_(node["value"], node["units"])

    def select_tag(self, obj, tags, ctx):
        tags = [tag for tag in tags if tag.startswith(WELDX_TAG_URI_BASE)]
        return sorted(tags)[-1]

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> list[int]:
        """Calculate the shape from static tagged tree instance."""
        if isinstance(node["value"], dict):  # ndarray
            return _get_instance_shape(node["value"])
        return [1]  # scalar


class PintUnitConverter(WeldxConverter):
    """A simple implementation of serializing a pint unit as tagged asdf node."""

    tags = ["asdf://weldx.bam.de/weldx/tags/units/units-0.1.*"]
    types = [
        "pint.util.Unit",  # pint >= 0.20
        "pint.unit.build_unit_class.<locals>.Unit",  # pint < 0.20
        "weldx.constants.U_",
    ]

    def to_yaml_tree(self, obj: pint.Unit, tag: str, ctx) -> str:
        """Convert to python dict."""
        return f"{obj:D}"  # use def/long (D) formatting for serialization

    def from_yaml_tree(self, node: str, tag: str, ctx) -> pint.Unit:
        """Reconstruct from tree."""
        return U_(node)
