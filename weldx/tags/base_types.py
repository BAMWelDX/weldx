"""Module providing ASDF implementations for basic python types."""
from uuid import UUID

from asdf.asdf import SerializationContext

from weldx.asdf.types import WeldxConverter

__all__ = ["UuidConverter"]


# UUID ---------------------------------------------------------------------------------
class UuidConverter(WeldxConverter):
    """Implements a version 4 UUID."""

    tags = ["asdf://weldx.bam.de/weldx/tags/uuid-0.1.*"]
    types = [UUID]

    def to_yaml_tree(self, obj: UUID, tag: str, ctx: SerializationContext) -> str:
        """Convert to python string."""
        return str(obj)

    def from_yaml_tree(self, node: str, tag: str, ctx: SerializationContext) -> UUID:
        """Reconstruct from string."""
        if tag.startswith("tag:weldx.bam.de:weldx"):  # legacy_code
            return UUID(node["uuid"])
        return UUID(node)
