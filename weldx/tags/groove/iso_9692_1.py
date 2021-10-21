"""ISO 9692-1 welding groove type definitions"""

from asdf.util import uri_match

from weldx.asdf.types import WeldxConverter
from weldx.asdf.util import get_weldx_extension
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove, _groove_name_to_type

__all__ = ["IsoGrooveConverter"]

_ISO_GROOVE_SCHEMA = "groove/iso_9692_1_2013_12/"


def _get_class_from_tag(instance_tag: str):
    groove_tag = instance_tag.rpartition("/iso_9692_1_2013_12/")[-1]
    return groove_tag.rpartition("-")[0]


class IsoGrooveConverter(WeldxConverter):
    """ASDF Groove type."""

    tags = ["asdf://weldx.bam.de/weldx/tags/groove/iso_9692_1_2013_12/*-0.1.*"]
    types = [IsoBaseGroove] + IsoBaseGroove.__subclasses__()

    def to_yaml_tree(self, obj: IsoBaseGroove, tag: str, ctx) -> dict:
        """Convert to python dict."""
        return obj.__dict__

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Convert from yaml node and tag to a groove."""
        groove_name = _get_class_from_tag(tag)
        groove = _groove_name_to_type[groove_name](**node)
        return groove

    def select_tag(self, obj, tags, ctx):
        """Select the highest supported new style tag according to groove name."""
        _snip = _ISO_GROOVE_SCHEMA + type(obj).__name__
        # select only new style tags
        selection = [tag for tag in self.tags if tag.startswith("asdf://")]
        # select the matching pattern
        selection = [tag for tag in selection if _snip in tag]
        if not len(selection) == 1:
            raise ValueError("Found multiple groove tags for selection.")

        ext = get_weldx_extension(ctx)
        tag = [t.tag_uri for t in ext.tags if uri_match(selection[0], t.tag_uri)]
        return tag[0]

    @classmethod
    def default_class_display_name(cls):
        """Return custom type string."""
        return "IsoGroove-Type"
