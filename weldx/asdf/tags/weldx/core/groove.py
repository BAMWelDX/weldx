"""<DOCSTRING>"""

import pint
from dataclasses import dataclass
from dataclasses import field
from typing import List
from asdf import yamlutil

from weldx import Q_
from weldx.asdf.types import WeldxType


def get_groove(groove_type, **kwargs):
    """<DEF DOCSTRING>"""
    if groove_type == "VGroove":
        return VGroove(**kwargs)
    if groove_type == "UGroove":
        return UGroove(**kwargs)
    if groove_type == "IGroove":
        return IGroove(**kwargs)
    if groove_type == "UVGroove":
        return UVGroove(**kwargs)


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
    h: pint.Quantity
    e: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(
        default_factory=lambda:
        ["2.12", "3.1.1", "3.1.2", "3.1.3", "4.1.1", "4.1.2", "4.1.3"]
    )


class GrooveType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/din_en_iso_9692-1_2013"
    version = "1.0.0"
    types = [VGroove, UGroove, UVGroove, IGroove]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        """<CLASS METHOD DOCSTRING>"""
        if isinstance(node, VGroove):
            components = dict(
                t=node.t,
                alpha=node.alpha,
                b=node.b,
                c=node.c,
            )
            code_number = yamlutil.custom_tree_to_tagged_tree(node.code_number, ctx)
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="SingleVGroove",
                code_number=code_number,
            )
            return tree

        if isinstance(node, UGroove):
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
