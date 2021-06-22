from dataclasses import dataclass

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

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


@asdf_dataclass_serialization
class ConnectionType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/connection"
    version = "1.0.0"
    types = [Connection]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
