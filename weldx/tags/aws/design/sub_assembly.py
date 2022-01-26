from __future__ import annotations

from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

from .connection import Connection
from .workpiece import Workpiece

__all__ = ["SubAssembly", "SubAssemblyConverter"]


@dataclass
class SubAssembly:
    """<CLASS DOCSTRING>"""

    workpiece: list[Workpiece]
    connection: Connection


SubAssemblyConverter = dataclass_serialization_class(
    class_type=SubAssembly,
    class_name="aws/design/sub_assembly",
    version="0.1.0",
)
