from dataclasses import dataclass

import pint

from weldx.asdf.util import dataclass_serialization_class

from .shielding_gas_type import ShieldingGasType

__all__ = ["ShieldingGasForProcedure", "ShieldingGasForProcedureConverter"]


@dataclass
class ShieldingGasForProcedure:
    """<CLASS DOCSTRING>"""

    use_torch_shielding_gas: bool
    torch_shielding_gas: ShieldingGasType
    torch_shielding_gas_flowrate: pint.Quantity
    use_backing_gas: bool = None
    backing_gas: ShieldingGasType = None
    backing_gas_flowrate: pint.Quantity = None
    use_trailing_gas: bool = None
    trailing_shielding_gas: ShieldingGasType = None
    trailing_shielding_gas_flowrate: pint.Quantity = None


ShieldingGasForProcedureConverter = dataclass_serialization_class(
    class_type=ShieldingGasForProcedure,
    class_name="aws/process/shielding_gas_for_procedure",
    version="0.1.0",
)
