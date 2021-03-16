"""Type aliases shared in visualization package."""
from typing import List, Union

import pandas as pd

types_time = Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]]

__all__ = ("types_time",)
