from weldx.asdf.types import WeldxType
from weldx.measurement import SignalSource

__all__ = ["SignalSource", "SignalSourceType"]


class SignalSourceType(WeldxType):
    """Serialization class for measurement sources."""

    name = "measurement/source"
    version = "1.0.0"
    types = [SignalSource]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: SignalSource, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = SignalSource(**tree)
        return obj
