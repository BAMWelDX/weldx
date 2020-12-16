from weldx.asdf.types import WeldxType
from weldx.measurement import GenericEquipment

__all__ = ["GenericEquipment", "GenericEquipmentType"]


class GenericEquipmentType(WeldxType):
    """Serialization class for generic-equipment."""

    name = "equipment/generic_equipment"
    version = "1.0.0"
    types = [GenericEquipment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: GenericEquipment, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        if "sources" not in tree:
            tree["sources"] = None
        obj = GenericEquipment(**tree)
        return obj
