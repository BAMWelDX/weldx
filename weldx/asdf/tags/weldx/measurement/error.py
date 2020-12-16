from weldx.asdf.types import WeldxType
from weldx.measurement import Error

__all__ = ["Error", "ErrorType"]


class ErrorType(WeldxType):
    """Serialization class for measurement errors."""

    name = "measurement/error"
    version = "1.0.0"
    types = [Error]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Error, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Error(**tree)
        return obj
