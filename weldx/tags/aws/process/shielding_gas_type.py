from __future__ import annotations

from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

from .gas_component import GasComponent

__all__ = ["ShieldingGasType", "ShieldingGasTypeConverter"]


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    gas_component: list[GasComponent]
    common_name: str
    designation: str = None


ShieldingGasTypeConverter = dataclass_serialization_class(
    class_type=ShieldingGasType,
    class_name="aws/process/shielding_gas_type",
    version="0.1.0",
)
