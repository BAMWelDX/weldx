from weldx.asdf.types import WeldxType
from weldx.measurement import SignalTransformation

__all__ = ["SignalTransformation", "SignalTransformationType"]


class SignalTransformationType(WeldxType):
    """Serialization class for data transformations."""

    name = "measurement/signal_transformation"
    version = "1.0.0"
    types = [SignalTransformation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: SignalTransformation, ctx):
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx) -> SignalTransformation:
        return SignalTransformation(**tree)
