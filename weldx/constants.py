"""Define constants for global library use."""
from pathlib import Path as _Path

from pint import UnitRegistry as _ureg

WELDX_PATH = _Path(__file__).parent.resolve()

WELDX_UNIT_REGISTRY = _ureg(
    preprocessors=[lambda string: string.replace("%", "percent")],  # allow %-sign
    force_ndarray_like=True,
)

# add percent unit
WELDX_UNIT_REGISTRY.define("percent = 0.01*count = %")
# swap plank constant for hour definition
WELDX_UNIT_REGISTRY.define("hour = 60*minute = h = hr")

WELDX_QUANTITY = WELDX_UNIT_REGISTRY.Quantity
Q_ = WELDX_QUANTITY
Q_.__name__ = "Q_"
Q_.__module_ = "pint.quantity"
Q_.__doc__ = """Create a quantity from a scalar or array.

The quantity class supports lots of physical units and will combine them during
mathematical operations

Examples
--------
>>> from weldx import Q_
>>> length = Q_(10, "mm")
>>> length

define a time:
>>> time = Q_(1, "s")
>>> time

lets combine length and time to get a velocity.
>>> v = length / time
>>> v
"""

__all__ = (
    "WELDX_PATH",
    "WELDX_QUANTITY",
    "WELDX_UNIT_REGISTRY",
    "Q_",
)
