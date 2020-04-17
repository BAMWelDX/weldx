"""<DOCSTRING>"""

import pint
from dataclasses import dataclass
from dataclasses import field
from typing import List

from weldx import Q_
from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree


def get_groove(
        groove_type,
        workpiece_thickness=None,
        workpiece_thickness2=None,
        root_gap=None,
        root_face=None,
        root_face2=None,
        root_face3=None,
        bevel_radius=None,
        bevel_angle=None,
        bevel_angle2=None,
        groove_angle=None,
        groove_angle2=None,
        special_depth=None,
):
    """
    Create a Groove from weldx.asdf.tags.weldx.core.groove.

    Make a selection from the given groove types.
    Groove Types:
        "VGroove"
        "UGroove"
        "IGroove"
        "UVGroove"
        "VVGroove"
        "HVGroove"
        "HUGroove"
        "DUGroove"
        "DHVGroove"
        "FrontalFaceGroove"

    Each groove type has a different set of attributes which are required. Only
    required attributes are considered. All the required attributes for Grooves
    are in Quantity values from pint and related units are accepted.
    Required Groove attributes:
        "VGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            alpha: groove angle
                - The groove angle is the whole angle of the V-Groove.
                  It is a pint Quantity in degree or radian.
            b: root gap
                - The root gap is the distance of the 2 workpieces. It can be 0 or None.
            c: root face
                - The root face is the length of the Y-Groove which is not part of the V.
                  It can be 0.

        "UGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            beta: bevel angle
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece.
            R: bevel radius
                - The bevel radius defines the length of the radius of the U-segment.
                  It is usually 6 millimeters.
            b: root gap
                - The root gap is the distance of the 2 workpieces. It can be 0 or None.
            c: root face
                - The root face is the height of the part below the U-segment.

    Example:
        from weldx import Q_ as Quantity
        from weldx.asdf.tags.weldx.core.groove import get_groove

        get_groove(groove_type="VGroove",
                   workpiece_thickness=Quantity(9, "mm"),
                   groove_angle=Quantity(50, "deg"),
                   root_face=Quantity(4, "mm"),
                   root_gap=Quantity(2, "mm"))

        get_groove(groove_type="UGroove",
                   workpiece_thickness=Quantity(15, "mm"),
                   bevel_angle=Quantity(9, "deg"),
                   bevel_radius=Quantity(6, "mm"),
                   root_face=Quantity(3, "mm"),
                   root_gap=Quantity(1, "mm"))

    :param groove_type: String specifying the Groove type
    :param workpiece_thickness: workpiece thickness
    :param workpiece_thickness2: workpiece thickness if type needs 2 thicknesses
    :param root_gap: root gap, gap between work pieces
    :param root_face: root face, upper part when 2 root faces are needed, middle part
                      when 3 are needed
    :param root_face2: root face, the lower part when 2 root faces are needed. upper
                       part when 3 are needed - used when min. 2 parts are needed
    :param root_face3: root face, usually the lower part - used when 3 parts are needed
    :param bevel_radius: bevel radius
    :param bevel_angle: bevel angle, usually the upper angle
    :param bevel_angle2: bevel angle, usually the lower angle
    :param groove_angle: groove angle, usually the upper angle
    :param groove_angle2: groove angle, usually the lower angle
    :param special_depth: special depth used for 4.1.2 Frontal-Face-Groove
    :return: an Groove from weldx.asdf.tags.weldx.core.groove
    """
    if groove_type == "VGroove":
        return VGroove(t=workpiece_thickness, alpha=groove_angle,
                       b=root_gap, c=root_face)

    if groove_type == "UGroove":
        return UGroove(t=workpiece_thickness, beta=bevel_angle,
                       R=bevel_radius, b=root_gap, c=root_face)

    if groove_type == "IGroove":
        return IGroove(t=workpiece_thickness, b=root_gap)

    if groove_type == "UVGroove":
        return UVGroove(t=workpiece_thickness, alpha=groove_angle, beta=bevel_angle,
                        R=bevel_radius, b=root_gap, h=root_face)

    if groove_type == "VVGroove":
        return VVGroove(t=workpiece_thickness, alpha=groove_angle, beta=bevel_angle,
                        b=root_gap, c=root_face, h=root_face2)

    if groove_type == "HVGroove":
        return HVGroove(t=workpiece_thickness, beta=bevel_angle,
                        b=root_gap, c=root_face)

    if groove_type == "HUGroove":
        return HUGroove(t=workpiece_thickness, beta=bevel_angle,
                       R=bevel_radius, b=root_gap, c=root_face)

    if groove_type == "DoubleVGroove":
        return DVGroove(t=workpiece_thickness, alpha_1=groove_angle, alpha_2=groove_angle2,
                        b=root_gap, c=root_face, h1=root_face2, h2=root_face3)

    if groove_type == "DoubleUGroove":
        return DUGroove(t=workpiece_thickness, beta_1=bevel_angle, beta_2=bevel_angle2,
                        h=root_face2, b=root_gap, c=root_face)

    if groove_type == "DoubleHVGroove":
        return DHVGroove(t=workpiece_thickness, beta_1=bevel_angle, beta_2=bevel_angle2,
                         h=root_face2, c=root_face, b=root_gap)

    if groove_type == "DoubleHUGroove":
        return DHUGroove(t=workpiece_thickness, beta_1=bevel_angle, beta_2=bevel_angle2,
                         R=bevel_radius, h=root_face2, c=root_face, b=root_gap)

    if groove_type == "FrontalFaceGroove":
        return FFGroove(t_1=workpiece_thickness, t_2=workpiece_thickness2,
                        alpha=groove_angle, b=root_gap, e=special_depth)


@dataclass
class VGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    alpha: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.3", "1.5"])


@dataclass
class VVGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    h: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.7"])


@dataclass
class UVGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    h: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.6"])


@dataclass
class UGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.8"])


@dataclass
class IGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.2.1", "1.2.2", "2.1"])


@dataclass
class HVGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.9.1", "1.9.2", "2.8"])


@dataclass
class HUGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.11", "2.10"])


# double Grooves
@dataclass
class DVGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    alpha_1: pint.Quantity
    alpha_2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.4", "2.5.1", "2.5.2"])


@dataclass
class DUGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    # t-c/2
    h: pint.Quantity
    c: pint.Quantity = Q_(3, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.7"])


@dataclass
class DHVGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    h: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.9.1", "2.9.2"])


@dataclass
class DHUGroove:
    """<CLASS DOCSTRING>"""
    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    R: pint.Quantity
    h: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.11"])


# Frontal Face - Groove
@dataclass
class FFGroove:
    """<CLASS DOCSTRING>"""
    t_1: pint.Quantity
    t_2: pint.Quantity
    alpha: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    e: pint.Quantity = None
    code_number: List[str] = field(
        default_factory=lambda:
        ["1.12", "1.13", "2.12", "3.1.1", "3.1.2", "3.1.3", "4.1.1", "4.1.2", "4.1.3"]
    )


class GrooveType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/din_en_iso_9692-1_2013"
    version = "1.0.0"
    types = [VGroove, VVGroove, UVGroove, UGroove, IGroove, UVGroove, HVGroove,
             HUGroove, DVGroove, DUGroove, DHVGroove, DHUGroove, FFGroove]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        """<CLASS METHOD DOCSTRING>"""
        if isinstance(node, VGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="SingleVGroove",
            )
            return tree

        if isinstance(node, UGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="SingleUGroove",
            )
            return tree

        if isinstance(node, UVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="UVGroove",
            )
            return tree

        if isinstance(node, IGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="IGroove",
            )
            return tree

        if isinstance(node, VVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="VVGroove",
            )
            return tree

        if isinstance(node, HVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="HVGroove",
            )
            return tree

        if isinstance(node, HUGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="HUGroove",
            )
            return tree

        if isinstance(node, DVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="DoubleVGroove",
            )
            return tree

        if isinstance(node, DUGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="DoubleUGroove",
            )
            return tree

        if isinstance(node, DHVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="DoubleHVGroove",
            )
            return tree

        if isinstance(node, DHUGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="DoubleHUGroove",
            )
            return tree

        if isinstance(node, FFGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type="FrontalFaceGroove",
            )
            return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """<CLASS METHOD DOCSTRING>"""
        if tree["type"] == "SingleVGroove":
            obj = VGroove(**tree["components"])
            return obj

        if tree["type"] == "SingleUGroove":
            obj = UGroove(**tree["components"])
            return obj

        if tree["type"] == "UVGroove":
            obj = UVGroove(**tree["components"])
            return obj

        if tree["type"] == "IGroove":
            obj = IGroove(**tree["components"])
            return obj

        if tree["type"] == "VVGroove":
            obj = VVGroove(**tree["components"])
            return obj

        if tree["type"] == "HVGroove":
            obj = HVGroove(**tree["components"])
            return obj

        if tree["type"] == "HUGroove":
            obj = HUGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleVGroove":
            obj = DVGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleUGroove":
            obj = DUGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleHVGroove":
            obj = DHVGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleHUGroove":
            obj = DHUGroove(**tree["components"])
            return obj

        if tree["type"] == "FrontalFaceGroove":
            obj = FFGroove(**tree["components"])
            return obj
