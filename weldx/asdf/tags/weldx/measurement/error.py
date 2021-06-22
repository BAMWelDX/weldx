from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization
from weldx.measurement import Error

__all__ = ["Error", "ErrorType"]


@asdf_dataclass_serialization
class ErrorType(WeldxType):
    """Serialization class for measurement errors."""

    name = "measurement/error"
    version = "1.0.0"
    types = [Error]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
