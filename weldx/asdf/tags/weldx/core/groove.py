"""<DOCSTRING>"""

import pint
from dataclasses import dataclass
from dataclasses import field
from typing import List
from asdf import yamlutil

from weldx import Q_
from weldx.asdf.types import WeldxType


def get_groove(
        groove_type,
        workpiece_thickness=None,
        workpiece_thickness2=None,
        root_gap=None,
        root_face=None,
        root_face2=None,
        bevel_radius=None,
        bevel_angle=None,
        bevel_angle2=None,
        groove_angle=None,
        groove_angle2=None,
        special_depth=None,
):
    """<DEF DOCSTRING>"""
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
                        b=root_gap, c=root_face)

    if groove_type == "HVGroove":
        return HVGroove(t=workpiece_thickness, beta=bevel_angle,
                        b=root_gap, c=root_face)

    if groove_type == "HUGroove":
        return HUGroove(t=workpiece_thickness, beta=bevel_angle,
                       R=bevel_radius, b=root_gap, c=root_face)

    if groove_type == "DoubleVGroove":
        return DVGroove(t=workpiece_thickness, alpha_1=groove_angle, alpha_2=groove_angle2,
                        b=root_gap, c=root_face)

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
            components = dict(t=node.t, alpha=node.alpha, b=node.b, c=node.c)
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="SingleVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, UGroove):
            components = dict(t=node.t, beta=node.beta, R=node.R, b=node.b, c=node.c)
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="SingleUGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, UVGroove):
            components = dict(
                t=node.t,
                alpha=node.alpha,
                beta=node.beta,
                R=node.R,
                b=node.b,
                h=node.h,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="UVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, IGroove):
            components = dict(
                t=node.t,
                b=node.b,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="IGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, VVGroove):
            components = dict(
                t=node.t,
                alpha=node.alpha,
                beta=node.beta,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="VVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, HVGroove):
            components = dict(
                t=node.t,
                beta=node.beta,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="HVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, HUGroove):
            components = dict(
                t=node.t,
                beta=node.beta,
                R=node.R,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="HUGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, DVGroove):
            components = dict(
                t=node.t,
                alpha_1=node.alpha_1,
                alpha_2=node.alpha_2,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="DoubleVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, DUGroove):
            components = dict(
                t=node.t,
                beta_1=node.beta_1,
                beta_2=node.beta_2,
                h=node.h,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="DoubleUGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, DHVGroove):
            components = dict(
                t=node.t,
                beta_1=node.beta_1,
                beta_2=node.beta_2,
                h=node.h,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="DoubleHVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, DHUGroove):
            components = dict(
                t=node.t,
                beta_1=node.beta_1,
                beta_2=node.beta_2,
                R=node.R,
                h=node.h,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="DoubleHUGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, FFGroove):
            components = dict(
                t_1=node.t_1,
                t_2=node.t_2,
                alpha=node.alpha,
                b=node.b,
                e=node.e,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="FrontalFaceGroove",
                code_number=code_number,
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
