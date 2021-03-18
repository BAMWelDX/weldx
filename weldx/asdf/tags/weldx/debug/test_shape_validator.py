from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pint

from weldx import Q_, TimeSeries
from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_shape_validator

__all__ = ["ShapeValidatorTestClass", "ShapeValidatorTestClassType"]


@dataclass
class ShapeValidatorTestClass:
    """Helper class to test the shape validator"""

    prop1: np.ndarray = np.ones((1, 2, 3))
    prop2: np.ndarray = np.ones((3, 2, 1))
    prop3: np.ndarray = np.ones((2, 4, 6, 8, 10))
    prop4: np.ndarray = np.ones((1, 3, 5, 7, 9))
    prop5: float = 3.141
    quantity: pint.Quantity = Q_(10, "m")
    timeseries: TimeSeries = TimeSeries(Q_(10, "m"))
    nested_prop: dict = field(
        default_factory=lambda: {
            "p1": np.ones((10, 8, 6, 4, 2)),
            "p2": np.ones((9, 7, 5, 3, 1)),
        }
    )
    time_prop: pd.DatetimeIndex = pd.timedelta_range("0s", freq="s", periods=9)
    optional_prop: np.ndarray = None


class ShapeValidatorTestClassType(WeldxType):
    """Helper class to test the shape validator"""

    name = "debug/test_shape_validator"
    version = "1.0.0"
    types = [ShapeValidatorTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {
        "wx_shape": wx_shape_validator,
    }

    @classmethod
    def to_tree(cls, node: ShapeValidatorTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShapeValidatorTestClass(**tree)
        return obj
