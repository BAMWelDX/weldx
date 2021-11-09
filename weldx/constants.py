"""Define constants for global library use."""
from pathlib import Path as _Path

import pint

UNIT_KEY = "units"  # default nomenclature for storing physical units information

WELDX_PATH = _Path(__file__).parent.resolve()

WELDX_UNIT_REGISTRY = pint.UnitRegistry(
    preprocessors=[
        lambda string: string.replace("%", "percent"),  # allow %-sign
        lambda string: string.replace("Δ°", "delta_deg"),  # parse Δ° for temperature
    ],
    force_ndarray_like=True,
)

# add percent unit
WELDX_UNIT_REGISTRY.define("percent = 0.01*count = %")
# swap plank constant for hour definition
WELDX_UNIT_REGISTRY.define("hour = 60*minute = h = hr")
# set default string format to short notation
# for more info on formatting: https://pint.readthedocs.io/en/stable/formatting.html
WELDX_UNIT_REGISTRY.default_format = "~"

WELDX_QUANTITY = WELDX_UNIT_REGISTRY.Quantity
Q_ = WELDX_QUANTITY
Q_.__name__ = "Q_"
Q_.__module_ = "pint.quantity"  # skipcq: PYL-W0212
Q_.__doc__ = """Create a quantity from a scalar or array.

The quantity class supports lots of physical units and will combine them during
mathematical operations.
For details on working with quantities and units, please see the
`pint documentation <https://pint.readthedocs.io/>`_

Examples
--------
>>> from weldx import Q_
>>> length = Q_(10, "mm")
>>> length
<Quantity(10, 'millimeter')>

define a time:

>>> time = Q_(1, "s")
>>> time
<Quantity(1, 'second')>

lets combine length and time to get a velocity.

>>> v = length / time
>>> v
<Quantity(10.0, 'millimeter / second')>
"""
__test__ = {"Q": Q_.__doc__}  # enable doctest checking.

U_ = WELDX_UNIT_REGISTRY.Unit
U_.__name__ = "U_"
U_.__module__ = "pint.unit"  # skipcq: PYL-W0212
U_.__doc__ = """For details on working with quantities and units, please see the
`pint documentation <https://pint.readthedocs.io/>`_
"""


__all__ = (
    "WELDX_PATH",
    "WELDX_QUANTITY",
    "WELDX_UNIT_REGISTRY",
    "Q_",
    "U_",
    "UNIT_KEY",
)
