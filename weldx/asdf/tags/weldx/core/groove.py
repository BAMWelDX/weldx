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


@dataclass
class VGroove:
    """<CLASS DOCSTRING>"""

    t: Q_
    alpha: Q_
    c: Q_ = Q_(0, "mm")
    b: Q_ = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.3", "1.5"])


@dataclass
class UGroove:
    """<CLASS DOCSTRING>"""

    t: Q_
    beta: Q_
    R: Q_
    c: Q_ = Q_(0, "mm")
    b: Q_ = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.8"])


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

    @classmethod
    def from_tree(cls, tree, ctx):
        if tree["type"] == "SingleVGroove":
            obj = VGroove(**tree["components"])
            return obj

        if tree["type"] == "SingleUGroove":
            obj = UGroove(**tree["components"])
            return obj
