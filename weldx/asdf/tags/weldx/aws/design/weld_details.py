from dataclasses import dataclass

import pint

from weldx.asdf.util import dataclass_serialization_class

__all__ = ["WeldDetails", "WeldDetailsType"]


@dataclass
class WeldDetails:
    """<CLASS DOCSTRING>"""

    joint_design: str
    weld_sizes: pint.Quantity
    number_of_passes: int


WeldDetailsType = dataclass_serialization_class(
    class_type=WeldDetails, class_name="aws/design/weld_details", version="1.0.0"
)
