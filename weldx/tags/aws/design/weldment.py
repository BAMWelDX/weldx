from __future__ import annotations

from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

from .sub_assembly import SubAssembly

__all__ = ["Weldment", "WeldmentConverter"]


@dataclass
class Weldment:
    """<CLASS DOCSTRING>"""

    sub_assembly: list[SubAssembly]


WeldmentConverter = dataclass_serialization_class(
    class_type=Weldment, class_name="aws/design/weldment", version="0.1.0"
)
