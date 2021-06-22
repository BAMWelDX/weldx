from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization
from weldx.measurement import Measurement

__all__ = ["Measurement", "MeasurementType"]


@asdf_dataclass_serialization
class MeasurementType(WeldxType):
    """Serialization class for measurement objects."""

    name = "measurement/measurement"
    version = "1.0.0"
    types = [Measurement]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
