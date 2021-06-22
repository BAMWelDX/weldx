from dataclasses import dataclass
from typing import List

from weldx.asdf.util import dataclass_serialization_class

from .gas_component import GasComponent

__all__ = ["ShieldingGasType", "ShieldingGasTypeType"]


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    gas_component: List[GasComponent]
    common_name: str
    designation: str = None


ShieldingGasTypeType = dataclass_serialization_class(
    class_type=ShieldingGasType,
    class_name="aws/process/shielding_gas_type",
    version="1.0.0",
)
