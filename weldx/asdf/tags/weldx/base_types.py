"""Module providing ASDF implementations for basic python types."""
from uuid import UUID

from asdf.extension import Converter

__all__ = ["UuidConverter"]


# UUID ---------------------------------------------------------------------------------
class UuidConverter(Converter):
    """Implements a version 4 UUID."""

    tags = ["asdf://weldx.bam.de/weldx/tags/uuid-1.*"]
    types = [UUID]

    @classmethod
    def to_yaml_tree(self, obj, tag, ctx):
        """Convert to python dict."""
        return dict(uuid=str(obj))

    @classmethod
    def from_yaml_tree(cls, node, tag, ctx):
        """Reconstruct from tree."""
        return UUID(node["uuid"])
