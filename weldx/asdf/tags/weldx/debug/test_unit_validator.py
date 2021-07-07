from dataclasses import dataclass, field

import numpy as np
import pint

from weldx import Q_
from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_unit_validator

__all__ = ["UnitValidatorTestClass", "UnitValidatorTestClassType"]


@dataclass
class UnitValidatorTestClass:
    """Testclass for validating pint.Quantities with wx_unit."""

    length_prop: pint.Quantity = Q_(1, "inch")
    velocity_prop: pint.Quantity = Q_(2, "km / s")
    current_prop: pint.Quantity = Q_(np.eye(2, 2), "mA")
    nested_prop: dict = field(
        default_factory=lambda: dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3"))
    )
    simple_prop: dict = field(default_factory=lambda: dict(value=float(3), unit="m"))


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
