"""Module providing ASDF implementations for basic python types."""

from uuid import UUID

import asdf

if asdf.__version__ >= "3.0.0":
    from asdf.extension import SerializationContext
else:
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
        return UUID(node)
