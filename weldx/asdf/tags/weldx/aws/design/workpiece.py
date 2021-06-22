from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

__all__ = ["Workpiece", "WorkpieceType"]


@dataclass
class Workpiece:
    """<CLASS DOCSTRING>"""

    geometry: str


WorkpieceType = dataclass_serialization_class(
    class_type=Workpiece, class_name="aws/design/workpiece", version="1.0.0"
)
