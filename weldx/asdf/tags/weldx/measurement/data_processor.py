from dataclasses import dataclass
from typing import List  # noqa: F401

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["DataProcessor", "DataProcessorType"]


@dataclass
class DataProcessor:
    """<TODO CLASS DOCSTRING>"""

    data: str


class DataProcessorType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "measurement/data_processor"
    version = "1.0.0"
    types = [DataProcessor]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: DataProcessor, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = DataProcessor(**tree)
        return obj
