from dataclasses import dataclass

import pandas as pd

from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_property_tag_validator

__all__ = ["PropertyTagTestClass", "PropertyTagTestClassType"]


@dataclass
class PropertyTagTestClass:
    """Helper class to test the shape validator"""

    prop1: pd.Timestamp = pd.Timestamp("2020-01-01")
    prop2: pd.Timestamp = pd.Timestamp("2020-01-02")
    prop3: pd.Timestamp = pd.Timestamp("2020-01-03")


class PropertyTagTestClassType(WeldxType):
    """Helper class to test the shape validator"""

    name = "debug/test_property_tag"
    version = "1.0.0"
    types = [PropertyTagTestClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {
        "wx_property_tag": wx_property_tag_validator,
    }

    @classmethod
    def to_tree(cls, node: PropertyTagTestClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = PropertyTagTestClass(**tree)
        return obj
