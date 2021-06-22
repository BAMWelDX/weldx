from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import Measurement

__all__ = ["Measurement", "MeasurementType"]


MeasurementType = dataclass_serialization_class(
    class_type=Measurement, class_name="measurement/measurement", version="1.0.0"
)
