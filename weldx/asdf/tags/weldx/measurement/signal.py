from weldx.asdf.types import WeldxType
from weldx.measurement import Signal

__all__ = ["Signal", "SignalType"]


class SignalType(WeldxType):
    """Serialization class for measurement signals."""

    name = "measurement/signal"
    version = "1.0.0"
    types = [Signal]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Signal, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        if "data" not in tree:
            tree["data"] = None
        obj = Signal(**tree)
        return obj
