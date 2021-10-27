from typing import List

import pint
from asdf.tagged import TaggedDict

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
        "pint.quantity.build_quantity_class.<locals>.Quantity",
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
        if tag.startswith("tag:weldx.bam.de:weldx"):  # legacy_code
            return Q_(node["value"], node["unit"])
        return Q_(node["value"], node["units"])

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> List[int]:
        """Calculate the shape from static tagged tree instance."""
        if isinstance(node["value"], dict):  # ndarray
            return _get_instance_shape(node["value"])
        return [1]  # scalar


class PintUnitConverter(WeldxConverter):
    """A simple implementation of serializing a pint unit as tagged asdf node."""

    tags = ["asdf://weldx.bam.de/weldx/tags/units/units-0.1.*"]
    types = [
        "pint.unit.build_unit_class.<locals>.Unit",
        "weldx.constants.U_",
    ]

    def to_yaml_tree(self, obj: pint.Unit, tag: str, ctx) -> str:
        """Convert to python dict."""
        from packaging import version

        if version.parse(pint.__version__) >= version.parse("0.18.0"):
            return f"{obj:D}"  # use def/long (D) formatting for serialization
        else:
            return f"{obj:.}"  # old long formatting

    def from_yaml_tree(self, node: str, tag: str, ctx) -> pint.Unit:
        """Reconstruct from tree."""
        return U_(node)
