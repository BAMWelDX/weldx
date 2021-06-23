from dataclasses import dataclass

import pint

from weldx.asdf.util import dataclass_serialization_class

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


JointPenetrationType = dataclass_serialization_class(
    class_type=JointPenetration,
    class_name="aws/design/joint_penetration",
    version="1.0.0",
)
