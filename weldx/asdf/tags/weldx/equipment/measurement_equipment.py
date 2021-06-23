from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import MeasurementEquipment

__all__ = ["MeasurementEquipment", "MeasurementEquipmentType"]


def _from_tree_mod(tree):
    if "sources" not in tree:
        tree["sources"] = None
    return tree


MeasurementEquipmentType = dataclass_serialization_class(
    class_type=MeasurementEquipment,
    class_name="equipment/measurement_equipment",
    version="1.0.0",
    from_tree_mod=_from_tree_mod,
)
