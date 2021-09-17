from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import SignalTransformation

__all__ = ["SignalTransformation", "SignalTransformationConverter"]


SignalTransformationConverter = dataclass_serialization_class(
    class_type=SignalTransformation,
    class_name="measurement/signal_transformation",
    version="0.1.0",
)
