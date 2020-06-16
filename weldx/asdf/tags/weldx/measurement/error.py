from dataclasses import dataclass
from typing import List  # noqa: F401

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["Error", "ErrorType"]


@dataclass
class Error:
    """<TODO CLASS DOCSTRING>"""

    data: str


class ErrorType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "measurement/error"
    version = "1.0.0"
    types = [Error]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Error, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Error(**tree)
        return obj
