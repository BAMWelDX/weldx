from dataclasses import dataclass
from typing import List  # noqa: F401

import numpy as np

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.asdf.validators import wx_shape_validator

__all__ = ["ShapeValidatorTestClass", "ShapeValidatorTestClassType"]


@dataclass
class ShapeValidatorTestClass:
    """Helper class to test the shape validator"""

    prop1: np.ndarray
    prop2: np.ndarray
    prop3: np.ndarray
    prop4: np.ndarray
    nested_prop: dict


class ShapeValidatorTestClassType(WeldxType):
    """Helper class to test the shape validator"""

    name = "debug/test_shape_validator"
    version = "1.0.0"
    types = [ShapeValidatorTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: ShapeValidatorTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ShapeValidatorTestClass(**tree)
        return obj


ShapeValidatorTestClassType.validators = {
    "wx_shape": wx_shape_validator,
}
