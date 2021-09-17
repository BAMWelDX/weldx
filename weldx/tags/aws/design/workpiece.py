from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

__all__ = ["Workpiece", "WorkpieceConverter"]


@dataclass
class Workpiece:
    """<CLASS DOCSTRING>"""

    geometry: str


WorkpieceConverter = dataclass_serialization_class(
    class_type=Workpiece, class_name="aws/design/workpiece", version="0.1.0"
)
