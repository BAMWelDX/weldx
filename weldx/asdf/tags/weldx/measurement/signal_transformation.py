from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization
from weldx.measurement import SignalTransformation

__all__ = ["SignalTransformation", "SignalTransformationType"]


@asdf_dataclass_serialization
class SignalTransformationType(WeldxType):
    """Serialization class for data transformations."""

    name = "measurement/signal_transformation"
    version = "1.0.0"
    types = [SignalTransformation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
