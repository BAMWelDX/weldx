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
        # a type or string can be used here
        # we use BOTH:
        # - the type: to allow pint to freely move the Quantity class path
        # - the string:
        #     to support weldx.constants.Q_ which reports itself as a
        #     'pint.Quantity'
        pint.Quantity,
        "pint.Quantity",
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
        # we need both the type and string (see the description
        # about pint.Quantity in the PintQuantityConverter above)
        pint.Unit,
        "pint.Unit",
        "weldx.constants.U_",
    ]

    def to_yaml_tree(self, obj: pint.Unit, tag: str, ctx) -> str:
        """Convert to python dict."""
        return f"{obj:D}"  # use def/long (D) formatting for serialization

    def from_yaml_tree(self, node: str, tag: str, ctx) -> pint.Unit:
        """Reconstruct from tree."""
        return U_(node)
