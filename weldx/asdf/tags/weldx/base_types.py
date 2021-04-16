"""Module providing ASDF implementations for basic python types."""
from uuid import UUID

from weldx.asdf.types import WeldxType


# UUID ---------------------------------------------------------------------------------
class UuidTypeASDF(WeldxType):
    """Implements a version 4 UUID."""

    name = "uuid"
    version = "1.0.0"
    types = [UUID]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: UUID, ctx):
        """convert to python dict"""
        return dict(uuid=str(node))

    @classmethod
    def from_tree(cls, tree, ctx):
        """Reconstruct form tree."""
        return UUID(tree["uuid"])
