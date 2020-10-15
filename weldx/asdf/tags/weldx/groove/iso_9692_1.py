"""ISO 9692-1 welding groove type definitions"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pint
from asdf.tagged import tag_object

import weldx.geometry as geo
from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.asdf.validators import wx_unit_validator
from weldx.constants import WELDX_QUANTITY as Q_


class IsoBaseGroove:
    """Generic base class for all groove types."""

    def parameters(self):
        """Return groove parameters as dictionary of quantities."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pint.Quantity)}

    def param_strings(self):
        """Generate string representation of parameters."""
        return [f"{k}={v:~}" for k, v in self.parameters().items()]


@dataclass
class IGroove(IsoBaseGroove):
    """An I-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t : pint.Quantity
        workpiece thickness
    b : pint.Quantity
        root gap
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.2.1", "1.2.2", "2.1"])


@dataclass
class VGroove(IsoBaseGroove):
    """A Single-V Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    alpha :
        groove angle
    b :
        root gap
    c :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    alpha: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.3", "1.5"])


_groove_type_to_name = {
    VGroove: "VGroove",
    # VVGroove: "VVGroove",
    # UVGroove: "UVGroove",
    # UGroove: "UGroove",
    IGroove: "IGroove",
    # HVGroove: "HVGroove",
    # HUGroove: "HUGroove",
    # DVGroove: "DVGroove",
    # DUGroove: "DUGroove",
    # DHVGroove: "DHVGroove",
    # DHUGroove: "DHUGroove",
    # FFGroove: "FFGroove",
}

_groove_name_to_type = {v: k for k, v in _groove_type_to_name.items()}


def _get_class_from_tag(instance_tag: str):
    groove_tag = instance_tag.rpartition("/iso_9692_1/")[-1]
    return groove_tag.rpartition("-")[0]


class IsoGrooveType(WeldxType):
    """ASDF Groove type."""

    name = ["groove/iso_9692_1/IGroove", "groove/iso_9692_1/VGroove"]
    version = "1.0.0"
    types = [
        IGroove,
        VGroove,
    ]
    requires = ["weldx"]
    validators = {"wx_unit": wx_unit_validator}

    @classmethod
    def to_tree(cls, node: IsoBaseGroove, ctx):
        """Convert tree and remove all None entries from node dictionary."""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def to_tree_tagged(cls, node: IsoBaseGroove, ctx):
        """Serialize tree with custom tag definition."""
        tree = cls.to_tree(node, ctx)
        tag = (
            "tag:weldx.bam.de:weldx/groove/iso_9692_1/"
            + type(node).__name__
            + "-"
            + str(cls.version)
        )
        return tag_object(tag, tree, ctx=ctx)

    @classmethod
    def from_tree_tagged(cls, tree, ctx):
        """Convert from tagged tree to a groove."""
        groove_name = _get_class_from_tag(tree._tag)
        groove = _groove_name_to_type[groove_name](**tree)
        return groove
