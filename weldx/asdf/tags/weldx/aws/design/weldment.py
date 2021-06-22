from dataclasses import dataclass
from typing import List

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

from .sub_assembly import SubAssembly

__all__ = ["Weldment", "WeldmentType"]


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: List[SubAssembly]


@asdf_dataclass_serialization
class WeldmentType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weldment"
    version = "1.0.0"
    types = [Weldment]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
