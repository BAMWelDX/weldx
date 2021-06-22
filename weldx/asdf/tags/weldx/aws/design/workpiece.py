from dataclasses import dataclass

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

__all__ = ["Workpiece", "WorkpieceType"]


@dataclass
class Workpiece:
    """<CLASS DOCSTRING>"""

    geometry: str


@asdf_dataclass_serialization
class WorkpieceType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/workpiece"
    version = "1.0.0"
    types = [Workpiece]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
