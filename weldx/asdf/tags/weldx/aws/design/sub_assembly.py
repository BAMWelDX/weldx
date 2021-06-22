from dataclasses import dataclass
from typing import List

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

from .connection import Connection
from .workpiece import Workpiece

__all__ = ["SubAssembly", "SubAssemblyType"]


@dataclass
class SubAssembly:
    """<CLASS DOCSTRING>"""

    workpiece: List[Workpiece]
    connection: Connection


@asdf_dataclass_serialization
class SubAssemblyType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/sub_assembly"
    version = "1.0.0"
    types = [SubAssembly]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
