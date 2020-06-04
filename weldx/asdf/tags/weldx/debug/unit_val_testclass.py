from dataclasses import dataclass
from typing import List  # noqa: F401

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.asdf.validators import validate_array_shape, validate_unit_dimension

__all__ = ["UnitValTestClass", "UnitValTestClassType"]


@dataclass
class UnitValTestClass:
    """<TODO CLASS DOCSTRING>"""

    length_prop: pint.Quantity
    velocity_prop: pint.Quantity
    current_prop: pint.Quantity
    nested_prop: dict


class UnitValTestClassType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "debug/unit_val_testclass"
    version = "1.0.0"
    types = [UnitValTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: UnitValTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = UnitValTestClass(**tree)
        return obj


UnitValTestClassType.validators = {
    "wx_shape": validate_array_shape,
    "wx_unit_validate": validate_unit_dimension,
}
