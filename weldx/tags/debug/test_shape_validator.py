from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pint

from weldx.asdf.util import dataclass_serialization_class
from weldx.constants import Q_
from weldx.core import TimeSeries

__all__ = ["ShapeValidatorTestClass", "ShapeValidatorTestClassConverter"]


@dataclass
class ShapeValidatorTestClass:
    """Helper class to test the shape validator"""

    prop1: np.ndarray = field(default_factory=lambda: np.ones((1, 2, 3)))
    prop2: np.ndarray = field(default_factory=lambda: np.ones((3, 2, 1)))
    prop3: np.ndarray = field(default_factory=lambda: np.ones((2, 4, 6, 8, 10)))
    prop4: np.ndarray = field(default_factory=lambda: np.ones((1, 3, 5, 7, 9)))
    prop5: float = 3.141
    quantity: pint.Quantity = Q_(10, "m")
    timeseries: TimeSeries = field(default_factory=lambda: TimeSeries(Q_(10, "m")))
    nested_prop: dict = field(
        default_factory=lambda: {
            "p1": np.ones((10, 8, 6, 4, 2)),
            "p2": np.ones((9, 7, 5, 3, 1)),
        }
    )
    time_prop: pd.DatetimeIndex = field(
        default_factory=lambda: pd.timedelta_range("0s", freq="s", periods=9)
    )
    optional_prop: np.ndarray = None


ShapeValidatorTestClassConverter = dataclass_serialization_class(
    class_type=ShapeValidatorTestClass,
    class_name="debug/test_shape_validator",
    version="0.1.0",
)
