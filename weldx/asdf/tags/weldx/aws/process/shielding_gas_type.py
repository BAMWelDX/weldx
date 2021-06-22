from dataclasses import dataclass
from typing import List

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization

from .gas_component import GasComponent

__all__ = ["ShieldingGasType", "ShieldingGasTypeType"]


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    gas_component: List[GasComponent]
    common_name: str
    designation: str = None


@asdf_dataclass_serialization
class ShieldingGasTypeType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/shielding_gas_type"
    version = "1.0.0"
    types = [ShieldingGasType]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
