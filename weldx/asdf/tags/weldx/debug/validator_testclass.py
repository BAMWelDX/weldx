from dataclasses import dataclass
from typing import List  # noqa: F401

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.asdf.validators import validate_array_shape, validate_unit_dimension

__all__ = ["ValidatorTestClass", "ValidatorTestClassType"]


@dataclass
class ValidatorTestClass:
    """<TODO CLASS DOCSTRING>"""

    length_prop: pint.Quantity
    velocity_prop: pint.Quantity
    current_prop: pint.Quantity
    nested_prop: dict


class ValidatorTestClassType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "debug/validator_testclass"
    version = "1.0.0"
    types = [ValidatorTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: ValidatorTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ValidatorTestClass(**tree)
        return obj


ValidatorTestClassType.validators = {
    "wx_shape": validate_array_shape,
    "wx_unit_validate": validate_unit_dimension,
}
