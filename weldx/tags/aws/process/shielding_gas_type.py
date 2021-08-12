from dataclasses import dataclass
from typing import List

from weldx.asdf.util import dataclass_serialization_class

from .gas_component import GasComponent

__all__ = ["ShieldingGasType", "ShieldingGasTypeConverter"]


@dataclass
class ShieldingGasType:
    """<CLASS DOCSTRING>"""

    # TODO: should be named gas_component__s__!!!
    gas_component: List[GasComponent]
    common_name: str
    designation: str = None

    def __post_init__(self):
        total_percentage = sum(g.gas_percentage for g in self.gas_component)
        if total_percentage != 100:
            raise ValueError("Gas components percentages do not sum to 100,"
                             f" but {total_percentage}")


ShieldingGasTypeConverter = dataclass_serialization_class(
    class_type=ShieldingGasType,
    class_name="aws/process/shielding_gas_type",
    version="1.0.0",
)
