from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import Measurement

__all__ = ["Measurement", "MeasurementConverter"]


MeasurementConverter = dataclass_serialization_class(
    class_type=Measurement, class_name="measurement/measurement", version="0.1.0"
)
