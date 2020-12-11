"""Welding process ASDF classes."""

from asdf.tagged import tag_object

from weldx.asdf.types import WeldxType
from weldx.welding.processes import GmawProcess


class GmawProcessTypeAsdf(WeldxType):
    """Custom serialization class for GmawProcess."""

    name = ["process/GMAW", "process/CLOOS/spray_arc", "process/CLOOS/pulse"]
    version = "1.0.0"
    types = [GmawProcess]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: GmawProcess, ctx):
        """Convert tree and remove all None entries from node dictionary."""
        tree = node.__dict__
        return tree

    @classmethod
    def to_tree_tagged(cls, node: GmawProcess, ctx):
        """Serialize tree with custom tag definition."""
        tree = cls.to_tree(node, ctx)
        tag = "tag:weldx.bam.de:weldx/process/" + tree["tag"] + "-" + str(cls.version)
        return tag_object(tag, tree, ctx=ctx)

    @classmethod
    def from_tree(cls, tree, ctx):
        """Read tree object into dataclass."""
        obj = GmawProcess(**tree)
        return obj
