"""ISO 9692-1 welding groove type definitions"""


from asdf.tagged import tag_object

from weldx.asdf.constants import WELDX_TAG_BASE
from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_unit_validator
from weldx.welding.groove.iso_9692_1 import IsoBaseGroove, _groove_name_to_type

_ISO_GROOVE_SCHEMA = "groove/iso_9692_1_2013_12/"


def _get_class_from_tag(instance_tag: str):
    groove_tag = instance_tag.rpartition("/iso_9692_1_2013_12/")[-1]
    return groove_tag.rpartition("-")[0]


class IsoGrooveType(WeldxType):
    """ASDF Groove type."""

    name = [_ISO_GROOVE_SCHEMA + g for g in _groove_name_to_type.keys()]
    version = "1.0.0"
    types = [IsoBaseGroove]
    requires = ["weldx"]
    validators = {"wx_unit": wx_unit_validator}

    @classmethod
    def to_tree(cls, node: IsoBaseGroove, ctx):
        """Convert tree and remove all None entries from node dictionary."""
        tree = node.__dict__
        return tree

    @classmethod
    def to_tree_tagged(cls, node: IsoBaseGroove, ctx):
        """Serialize tree with custom tag definition."""
        tree = cls.to_tree(node, ctx)
        tag = (
            WELDX_TAG_BASE
            + "/"
            + _ISO_GROOVE_SCHEMA
            + type(node).__name__
            + "-"
            + str(cls.version)
        )
        return tag_object(tag, tree, ctx=ctx)

    @classmethod
    def from_tree_tagged(cls, tree, ctx):
        """Convert from tagged tree to a groove."""
        groove_name = _get_class_from_tag(tree._tag)
        groove = _groove_name_to_type[groove_name](**tree)
        return groove
