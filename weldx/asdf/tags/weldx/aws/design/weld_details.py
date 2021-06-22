from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

__all__ = ["WeldDetails", "WeldDetailsType"]


@dataclass
class WeldDetails:
    """<CLASS DOCSTRING>"""

    joint_design: str
    weld_sizes: pint.Quantity
    number_of_passes: int


@asdf_dataclass_serialization
class WeldDetailsType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/weld_details"
    version = "1.0.0"
    types = [WeldDetails]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
