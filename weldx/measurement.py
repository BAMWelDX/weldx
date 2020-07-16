"""Contains measurement related classes and functions."""

from dataclasses import dataclass, field
from typing import List, Union  # noqa: F401

import numpy as np
import pandas as pd
import pint
import xarray as xr

import weldx.utility as ut
from weldx.asdf.tags.weldx.core.mathematical_expression import MathematicalExpression


# measurement --------------------------------------------------------------------------
@dataclass
class Data:
    """Simple dataclass implementation for measurement data."""

    name: str
    data: xr.DataArray


@dataclass
class Error:
    """Simple dataclass implementation for signal transformation errors."""

    deviation: float


@dataclass
class Signal:
    """Simple dataclass implementation for measurement signals."""

    signal_type: str
    unit: str
    data: Union[Data, None]


@dataclass
class DataTransformation:
    """Simple dataclass implementation for signal transformations."""

    name: str
    input_signal: Signal
    output_signal: Signal
    error: Error
    func: str = None
    meta: str = None


@dataclass
class Source:
    """Simple dataclass implementation for signal sources."""

    name: str
    output_signal: Signal
    error: Error


@dataclass
class MeasurementChain:
    """Simple dataclass implementation for measurement chains."""

    name: str
    data_source: Source
    data_processors: List = field(default_factory=lambda: [])


@dataclass
class Measurement:
    """Simple dataclass implementation for generic measurements."""

    name: str
    data: Data
    measurement_chain: MeasurementChain


class TimeSeries:
    """Describes a the behaviour of a quantity in time."""

    def __init__(self, data, time=None, interpolation=None):
        self._time = None
        self._data = None
        self._interpolation = None

        if isinstance(data, pint.Quantity):
            if isinstance(data.magnitude, np.ndarray):
                # TODO: check interpolation type (constant, linear, etc.)
                if interpolation is None:
                    raise ValueError(
                        "An interpolation method must be specified "
                        "if discrete values are used."
                    )

                self._data = xr.DataArray(
                    data=data, dims=["time"], coords={"time": time}
                )
                self._interpolation = interpolation
            else:
                # TODO: check if time is None or time range
                self._time = time
                self._data = data
        elif isinstance(data, MathematicalExpression):
            pass
        else:
            raise ValueError(f'The data type "{type(data)}" is not supported.')

    @property
    def data(self):
        if isinstance(self._data, xr.DataArray):
            return self._data.data
        return self._data

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def time(self):
        if isinstance(self._data, xr.DataArray):
            return ut.to_pandas_time_index(self._data.time.data)
        return self._time

    @classmethod
    def from_discrete_values(
        cls, time: pd.TimedeltaIndex, data: pint.Quantity, interpolation: str
    ):
        """
        Create a time series from discrete timestamps/time deltas and data points.

        Parameters
        ----------
        time
        data
        interpolation

        Returns
        -------

        """
        return cls(time=time, values=data, interpolation=interpolation)

    @classmethod
    def from_mathematical_expression(cls):
        pass


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """Simple dataclass implementation for generic equipment."""

    name: str
    sources: List = field(default_factory=lambda: [])
    data_transformations: List = field(default_factory=lambda: [])
