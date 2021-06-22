from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

from .joint_penetration import JointPenetration
from .weld_details import WeldDetails

__all__ = ["Connection", "ConnectionType"]


@dataclass
class Connection:
    """<CLASS DOCSTRING>"""

    joint_type: str
    weld_type: str
    joint_penetration: JointPenetration
    weld_details: WeldDetails


ConnectionType = dataclass_serialization_class(
    class_type=Connection,
    class_name="aws/design/connection",
    version="1.0.0",
)
