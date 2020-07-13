from dataclasses import dataclass

import pandas as pd

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.asdf.validators import wx_property_tag_validator

__all__ = ["TestPropertyTagClass", "TestPropertyTagClassType"]


@dataclass
class TestPropertyTagClass:
    """Helper class to test the shape validator"""

    prop1: pd.Timestamp = pd.Timestamp("2020-01-01")
    prop2: pd.Timestamp = pd.Timestamp("2020-01-02")
    prop3: pd.Timestamp = pd.Timestamp("2020-01-03")


class TestPropertyTagClassType(WeldxType):
    """Helper class to test the shape validator"""

    name = "debug/test_property_tag"
    version = "1.0.0"
    types = [TestPropertyTagClass]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: TestPropertyTagClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = TestPropertyTagClass(**tree)
        return obj


TestPropertyTagClassType.validators = {
    "wx_property_tag": wx_property_tag_validator,
}
