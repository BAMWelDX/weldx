from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization
from weldx.measurement import SignalSource

__all__ = ["SignalSource", "SignalSourceType"]


@asdf_dataclass_serialization
class SignalSourceType(WeldxType):
    """Serialization class for measurement sources."""

    name = "measurement/source"
    version = "1.0.0"
    types = [SignalSource]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
