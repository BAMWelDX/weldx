from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import MeasurementEquipment

__all__ = ["MeasurementEquipment", "MeasurementEquipmentConverter"]


def _from_yaml_tree_mod(tree):
    if "sources" not in tree:
        tree["sources"] = None
    return tree


MeasurementEquipmentConverter = dataclass_serialization_class(
    class_type=MeasurementEquipment,
    class_name="equipment/measurement_equipment",
    version="0.1.0",
    from_yaml_tree_mod=_from_yaml_tree_mod,
)
