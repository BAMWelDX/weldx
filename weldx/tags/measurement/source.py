from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import SignalSource

__all__ = ["SignalSource", "SignalSourceConverter"]


SignalSourceConverter = dataclass_serialization_class(
    class_type=SignalSource, class_name="measurement/source", version="1.0.0"
)
