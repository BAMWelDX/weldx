from dataclasses import dataclass
from typing import List

from weldx.asdf.util import dataclass_serialization_class

from .sub_assembly import SubAssembly

__all__ = ["Weldment", "WeldmentType"]


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: List[SubAssembly]


WeldmentType = dataclass_serialization_class(
    class_type=Weldment, class_name="aws/design/weldment", version="1.0.0"
)
