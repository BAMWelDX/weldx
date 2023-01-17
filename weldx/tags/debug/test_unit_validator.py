from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pint
import xarray as xr

from weldx.asdf.util import dataclass_serialization_class
from weldx.constants import Q_
from weldx.core import TimeSeries

__all__ = ["UnitValidatorTestClass", "UnitValidatorTestClassConverter"]


@dataclass
class UnitValidatorTestClass:
    """Testclass for validating pint.Quantities with wx_unit."""

    length_prop: pint.Quantity = Q_(1, "m")
    velocity_prop: pint.Quantity = Q_(2, "km / s")
    current_prop: pint.Quantity = Q_(np.eye(2, 2), "mA")
    nested_prop: dict = field(
        default_factory=lambda: dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3"))
    )
    simple_prop: dict = field(default_factory=lambda: dict(value=float(3), units="m"))
    delta_prop: dict = Q_(100, "Δ°C")
    dimensionless: Union[float, int, np.ndarray, pint.Quantity, xr.DataArray] = 3.14
    custom_object: Any = field(
        default_factory=lambda: TimeSeries(Q_([0, 5], "A"), Q_([0, 1], "s"))
    )


UnitValidatorTestClassConverter = dataclass_serialization_class(
    class_type=UnitValidatorTestClass,
    class_name="debug/test_unit_validator",
    version="0.1.0",
)
