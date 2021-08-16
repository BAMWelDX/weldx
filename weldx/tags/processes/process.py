"""Welding process ASDF classes."""

from weldx.asdf.types import WeldxConverter, format_tag
from weldx.welding.processes import GmawProcess

__all__ = ["GmawProcessConverter"]


class GmawProcessConverter(WeldxConverter):
    """Custom serialization class for GmawProcess."""

    tags = [
        "asdf://weldx.bam.de/weldx/tags/process/GMAW-1.0.0",
        "asdf://weldx.bam.de/weldx/tags/process/CLOOS/spray_arc-1.0.0",
        "asdf://weldx.bam.de/weldx/tags/process/CLOOS/pulse-1.0.0",
    ]
    types = [GmawProcess]

    def to_yaml_tree(self, obj: GmawProcess, tag: str, ctx) -> dict:
        """Convert to python dict."""
        return obj.__dict__

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Convert from yaml node."""
        return GmawProcess(**node)

    def select_tag(self, obj: GmawProcess, tags, ctx):
        """Select new style tag according to groove name."""
        tag = format_tag(tag_name="process/" + obj.tag, version="1.0.0")
        if tag not in self.tags:
            raise ValueError("The generated process tag is not supported")
        return tag
