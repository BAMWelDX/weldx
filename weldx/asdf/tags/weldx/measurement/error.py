from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import Error

__all__ = ["Error", "ErrorType"]


ErrorType = dataclass_serialization_class(
    class_type=Error, class_name="measurement/error", version="1.0.0"
)
