"""Module providing ASDF implementations for basic python types."""
from uuid import UUID

from asdf.asdf import SerializationContext

from weldx.asdf.types import WeldxConverter

__all__ = ["UuidConverter"]


# UUID ---------------------------------------------------------------------------------
class UuidConverter(WeldxConverter):
    """Implements a version 4 UUID."""

    tags = ["asdf://weldx.bam.de/weldx/tags/uuid-1.*"]
    types = [UUID]

    def to_yaml_tree(self, obj: UUID, tag: str, ctx: SerializationContext) -> dict:
        """Convert to python dict."""
        return dict(uuid=str(obj))

    def from_yaml_tree(self, node: dict, tag: str, ctx: SerializationContext):
        """Reconstruct from tree."""
        return UUID(node["uuid"])
