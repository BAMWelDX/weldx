"""Type aliases shared in visualization package."""
from typing import List, Tuple, Union

import pandas as pd

types_time = Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]]
types_limits = Union[List[Tuple[float, float]], Tuple[float, float]]

__all__ = ("types_time", "types_limits")
