from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

__all__ = ["JointPenetration", "JointPenetrationType"]


@dataclass
class JointPenetration:
    """<CLASS DOCSTRING>"""

    complete_or_partial: str
    root_penetration: pint.Quantity
    groove_weld_size: float = None
    incomplete_joint_penetration: float = None
    weld_size: float = None
    weld_size_E1: float = None
    weld_size_E2: float = None
    depth_of_fusion: float = None


@asdf_dataclass_serialization
class JointPenetrationType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/joint_penetration"
    version = "1.0.0"
    types = [JointPenetration]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
