"""ISO 9692-1 welding groove type definitions"""


from asdf.tagged import tag_object

from weldx.asdf.constants import WELDX_TAG_BASE
from weldx.asdf.types import WeldxConverter, format_tag
from weldx.asdf.validators import wx_unit_validator
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove, _groove_name_to_type

__all__ = ["IsoGrooveConverter"]

_ISO_GROOVE_SCHEMA = "groove/iso_9692_1_2013_12/"


def _get_class_from_tag(instance_tag: str):
    groove_tag = instance_tag.rpartition("/iso_9692_1_2013_12/")[-1]
    return groove_tag.rpartition("-")[0]


class IsoGrooveConverter(WeldxConverter):
    """ASDF Groove type."""

    tags = [
        format_tag(tag_name=_ISO_GROOVE_SCHEMA + g, version="1.0.0")  # TODO: check 1.*
        for g in _groove_name_to_type.keys()
    ]
    types = [cls for cls in IsoBaseGroove.__subclasses__()]
    validators = {"wx_unit": wx_unit_validator}

    @classmethod
    def to_yaml_tree(self, obj: IsoBaseGroove, tag: str, ctx):
        """Convert to tree."""
        return obj.__dict__

    @classmethod
    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Convert from yaml node and tag to a groove."""
        groove_name = _get_class_from_tag(tag)
        groove = _groove_name_to_type[groove_name](**node)
        return groove

    def select_tag(self, obj, tags, ctx):
        """Select new style tag according to groove name."""
        _snip = _ISO_GROOVE_SCHEMA + type(obj).__name__
        selection = [tag for tag in self.tags if tag.startswith("asdf://")]
        selection = [tag for tag in selection if _snip in tag]
        assert len(selection) == 1
        return selection[0]
