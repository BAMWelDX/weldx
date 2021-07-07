from dataclasses import dataclass, field

import numpy as np
import pint

from weldx import Q_
from weldx.asdf.util import dataclass_serialization_class
from weldx.asdf.validators import wx_unit_validator

__all__ = ["UnitValidatorTestClass", "UnitValidatorTestClassType"]


@dataclass
class UnitValidatorTestClass:
    """Testclass for validating pint.Quantities with wx_unit."""

    length_prop: pint.Quantity = Q_(1, "m")
    velocity_prop: pint.Quantity = Q_(2, "km / s")
    current_prop: pint.Quantity = Q_(np.eye(2, 2), "mA")
    nested_prop: dict = field(
        default_factory=lambda: dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3"))
    )
    simple_prop: dict = field(default_factory=lambda: dict(value=float(3), unit="m"))


UnitValidatorTestClassType = dataclass_serialization_class(
    class_type=UnitValidatorTestClass,
    class_name="debug/test_unit_validator",
    version="1.0.0",
    validators={
        "wx_unit": wx_unit_validator,
    },
)
