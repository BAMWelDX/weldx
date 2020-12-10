from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_unit_validator

__all__ = ["UnitValidatorTestClass", "UnitValidatorTestClassType"]


@dataclass
class UnitValidatorTestClass:
    """Testclass for validating pint.Quantities with wx_unit."""

    length_prop: pint.Quantity
    velocity_prop: pint.Quantity
    current_prop: pint.Quantity
    nested_prop: dict
    simple_prop: dict


class UnitValidatorTestClassType(WeldxType):
    """Serialization testclass custom validators."""

    name = "debug/test_unit_validator"
    version = "1.0.0"
    types = [UnitValidatorTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {
        "wx_unit": wx_unit_validator,
    }

    @classmethod
    def to_tree(cls, node: UnitValidatorTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = UnitValidatorTestClass(**tree)
        return obj
