"""Contains the serialization class for the weldx.core.TimeSeries."""
from typing import List

import numpy as np
import pint
from asdf.tagged import TaggedDict

from weldx.asdf.types import WeldxConverter
from weldx.constants import Q_, U_
from weldx.core import TimeSeries


class TimeSeriesConverter(WeldxConverter):
    """Serialization class for weldx.core.TimeSeries"""

    name = "core/time_series"
    version = "0.1.0"
    types = [TimeSeries]

    def to_yaml_tree(self, obj: TimeSeries, tag: str, ctx) -> dict:
        """Convert to python dict."""
        if isinstance(obj.data, pint.Quantity):
            if obj.shape == tuple([1]):  # constant
                return {
                    "units": obj.units,
                    "value": obj.data.magnitude[0],
                }
            return {
                "time": obj.time.as_pandas_index(),
                "units": obj.units,
                "shape": obj.shape,
                "interpolation": obj.interpolation,
                "values": obj.data.magnitude,
            }
        return {"expression": obj.data, "units": obj.units, "shape": obj.shape}

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        if "value" in node:  # constant
            if "units" in node:
                _units = U_(node["units"])
            else:  # legacy_code
                _units = U_(node["unit"])
            values = Q_(np.asarray(node["value"]), _units)
            return TimeSeries(values)
        if "values" in node:
            time = node["time"]
            interpolation = node["interpolation"]
            if "units" in node:
                _units = U_(node["units"])
            else:  # legacy_code
                _units = U_(node["unit"])
            values = Q_(node["values"], _units)
            return TimeSeries(values, time, interpolation)

        return TimeSeries(node["expression"])  # mathexpression

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> List[int]:
        """Calculate the shape from static tagged tree instance."""
        if "shape" in node:  # this should not be reached but lets make sure
            return node["shape"]
        return [1]  # scalar
