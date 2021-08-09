"""Module providing ASDF implementations for basic python types."""
from uuid import UUID

from asdf.extension import Converter

from weldx.asdf._types import WeldxConverterMeta

__all__ = ["UuidConverter"]


# UUID ---------------------------------------------------------------------------------
class UuidConverter(Converter, metaclass=WeldxConverterMeta):
    """Implements a version 4 UUID."""

    tags = ["asdf://weldx.bam.de/weldx/tags/uuid-1.*"]
    types = [UUID]

    @classmethod
    def to_yaml_tree(self, obj, tag, ctx):
        """Convert to python dict."""
        return dict(uuid=str(obj))

    @classmethod
    def from_yaml_tree(self, node, tag, ctx):
        """Reconstruct from tree."""
        return UUID(node["uuid"])
