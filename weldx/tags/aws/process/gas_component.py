from dataclasses import dataclass

import pint

from weldx.asdf.util import dataclass_serialization_class

__all__ = ["GasComponent", "GasComponentConverter"]


@dataclass
class GasComponent:
    """<CLASS DOCSTRING>"""

    gas_chemical_name: str
    gas_percentage: pint.Quantity


GasComponentConverter = dataclass_serialization_class(
    class_type=GasComponent, class_name="aws/process/gas_component", version="0.1.0"
)
