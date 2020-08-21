"""Temporary module for welding related classes."""

from dataclasses import dataclass
from typing import Dict

from asdf.tagged import tag_object

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.core import TimeSeries


@dataclass
class GmawProcess:
    """Container class for all GMAW processes."""

    base_process: str
    manufacturer: str
    power_source: str
    parameters: Dict[str, TimeSeries]
    tag: str = None
    meta: dict = None

    def __post_init__(self):
        """Set defaults and convert parmater inputs."""
        if self.tag is None:
            self.tag = "GMAW"

        self.parameters = {
            k: (v if isinstance(v, TimeSeries) else TimeSeries(v))
            for k, v in self.parameters.items()
        }


class GmawProcessTypeAsdf(WeldxType):
    """Custom serialization class for GmawProcess."""

    name = ["process/GMAW", "process/CLOOS/spray_arc", "process/CLOOS/pulse"]
    version = "1.0.0"
    types = [GmawProcess]
    requires = ["weldx"]

    @classmethod
    def to_tree(cls, node: GmawProcess, ctx):
        """Convert tree and remove all None entries from node dictionary."""
        tree = drop_none_attr(node)
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
