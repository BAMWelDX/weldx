from dataclasses import dataclass
from dataclasses import field
from typing import List
from asdf import yamlutil
from weldx import Q_
from weldx.asdf.types import WeldxType


@dataclass
class VGroove:
    """<CLASS DOCSTRING>"""
    t: Q_
    alpha: Q_
    c: Q_ = Q_(0, "mm")
    b: Q_ = Q_(0, "mm")


@dataclass
class UGroove:
    """<CLASS DOCSTRING>"""
    t: Q_
    beta: Q_
    R: Q_
    c: Q_ = Q_(0, "mm")
    b: Q_ = Q_(0, "mm")


class GrooveType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/din_en_iso_9692-1_2013"
    version = "1.0.0"
    types = [VGroove, UGroove]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        if isinstance(node, VGroove):
            components = node.__dict__
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="SingleVGroove",
            )
            return tree

        if isinstance(node, UGroove):
            components = node.__dict__
            tree = dict(
                components=yamlutil.custom_tree_to_tagged_tree(components, ctx),
                type="SingleUGroove"
            )
            return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        if tree["type"] == "SingleVGroove":
            obj = VGroove(**tree["components"])
            return obj

        if tree["type"] == "SingleUGroove":
            obj = UGroove(**tree["components"])
            return obj
