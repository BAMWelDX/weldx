from weldx.asdf.types import WeldxType
from weldx.measurement import MeasurementEquipment

__all__ = ["MeasurementEquipment", "MeasurementEquipmentType"]


class MeasurementEquipmentType(WeldxType):
    """Serialization class for generic-equipment."""

    name = "equipment/measurement_equipment"
    version = "1.0.0"
    types = [MeasurementEquipment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: MeasurementEquipment, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        if "sources" not in tree:
            tree["sources"] = None
        obj = MeasurementEquipment(**tree)
        return obj
