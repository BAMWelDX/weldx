"""Contains classes and functions related to time."""

from __future__ import annotations

from functools import reduce
from typing import List, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp
from pandas.api.types import is_object_dtype
from xarray import DataArray

from weldx.types import types_time_like, types_timestamp_like

from .util import pandas_time_delta_to_quantity

__all__ = ["Time"]

# list of types that are supported to be stored in Time._time
_data_base_types = (pd.Timedelta, pd.Timestamp, pd.DatetimeIndex, pd.TimedeltaIndex)


class Time:
    """Provides a unified interface for time related operations."""

    def __init__(
        self,
        time: Union[types_time_like, Time],
        time_ref: Union[types_timestamp_like, Time, None] = None,
    ):
        """Initialize a new `Time` class.

        Parameters
        ----------
        time :
            A supported class that represents either absolute or relative times. The
            data must be in ascending order.
        time_ref :
            An absolute reference point in time (timestamp). The return values of all
            accessors that return relative times will be calculated towards this
            reference time.
            This will turn the data of this class into absolute time values if relative
            times are passed as ``time`` parameter. In case ``time`` already contains
            absolute values and this parameter is set to ``None``, the first value of
            the data will be used as reference time.

        Raises
        ------
        ValueError:
            When time values passed are not sorted in monotonic increasing order.

        """
        if isinstance(time, Time):
            time_ref = time_ref if time_ref is not None else time._time_ref
            time = time._time

        if isinstance(time, _data_base_types):
            pass
        elif isinstance(time, pint.Quantity):
            time = Time._convert_quantity(time)
        elif isinstance(time, (xr.DataArray, xr.Dataset)):
            time = Time._convert_xarray(time)
        else:
            time = Time._convert_other(time)

        # catch scalar Index-objects
        if isinstance(time, pd.Index) and len(time) == 1:
            time = time[0]

        # sanity check
        if not isinstance(time, _data_base_types):
            raise TypeError("Could not create pandas time-like object.")

        if time_ref is not None:
            time_ref = pd.Timestamp(time_ref)
            if isinstance(time, pd.Timedelta):
                time = time + time_ref

        if isinstance(time, pd.TimedeltaIndex) & (time_ref is not None):
            time = time + time_ref

        if isinstance(time, pd.Index) and not time.is_monotonic_increasing:
            raise ValueError("The time values passed are not monotonic increasing.")

        self._time = time
        self._time_ref = time_ref

    def __add__(self, other: Union[types_time_like, Time]) -> Time:
        """Element-wise addition between `Time` object and compatible types."""
        return Time(time=self._time + Time(other).as_pandas())

    def __radd__(self, other: Union[types_time_like, Time]) -> Time:
        """Element-wise addition between `Time` object and compatible types."""
        return self + other

    def __sub__(self, other: Union[types_time_like, Time]) -> Time:
        """Element-wise subtraction between `Time` object and compatible types."""
        return Time(time=self._time - Time(other).as_pandas())

    def __rsub__(self, other: Union[types_time_like, Time]) -> Time:
        """Element-wise subtraction between `Time` object and compatible types."""
        return Time(time=Time(other).as_pandas() - self._time)

    def __eq__(self, other: Union[types_time_like, Time]) -> Union[bool, List[bool]]:
        """Element-wise comparisons between time object and compatible types.

        See Also
        --------
        equals : Check equality of `Time` objects.
        """
        return self._time == Time(other).as_pandas()

    def equals(self, other: Time) -> bool:
        """Test for matching ``time`` and ``reference_time`` between objects."""
        return np.all(self._time == other._time) & (self._time_ref == other._time_ref)

    def all_close(self, other: Union[types_time_like, Time]) -> bool:
        """Return `True` if another object compares equal within a certain tolerance."""
        # TODO: handle tolerances ?
        return np.allclose(self._time, Time(other).as_pandas())

    def as_quantity(self) -> pint.Quantity:
        """Return the data as `pint.Quantity`."""
        if self.is_absolute:
            q = pandas_time_delta_to_quantity(self._time - self.reference_time)
            setattr(q, "time_ref", self.reference_time)  # store time_ref info
            return q
        return pandas_time_delta_to_quantity(self._time)

    def as_timedelta(self) -> Union[Timedelta, TimedeltaIndex]:
        """Return the data as `pandas.TimedeltaIndex`."""
        if self.is_absolute:
            return self._time - self.reference_time
        return self._time

    def as_datetime(self) -> Union[Timestamp, DatetimeIndex]:
        """Return the data as `pandas.DatetimeIndex`."""
        if not self.is_absolute:
            raise TypeError("Cannot convert non absolute Time object to datetime")
        return self._time

    def as_pandas(
        self,
    ) -> Union[pd.Timedelta, pd.TimedeltaIndex, pd.Timestamp, pd.DatetimeIndex]:
        """Return the underlying pandas time datatype."""
        return self._time

    def as_pandas_index(self) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
        """Return a pandas index type regardless of length.

        This is useful when using time as coordinate in xarray types.
        """
        if isinstance(self._time, pd.Timestamp):
            return pd.DatetimeIndex([self._time])
        if isinstance(self._time, pd.Timedelta):
            return pd.TimedeltaIndex([self._time])
        return self._time

    def as_data_array(self) -> DataArray:
        """Return the data as `xarray.DataArray`."""
        da = xr.DataArray(self._time, coords={"time": self._time}, dims=["time"])
        da.time.attrs["time_ref"] = self.reference_time
        return da

    @property
    def reference_time(self) -> Union[Timestamp, None]:
        """Get the reference time."""
        if isinstance(self._time, DatetimeIndex):
            return self._time_ref if self._time_ref is not None else self._time[0]
        if isinstance(self._time, Timestamp):
            return self._time_ref if self._time_ref is not None else self._time
        return None

    @reference_time.setter
    def reference_time(self, time_ref: Union[types_timestamp_like, Time]):
        """Set the reference time."""
        self._time_ref = pd.Timestamp(time_ref)

    @property
    def is_absolute(self) -> bool:
        """Return `True` if the class has a reference time and `False` otherwise."""
        return isinstance(self._time, (Timestamp, DatetimeIndex))

    @property
    def length(self) -> int:
        """Return the length of the data."""
        if isinstance(self._time, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            return len(self._time)
        return 1

    @property
    def is_timestamp(self) -> bool:
        """Return `True` if the data represents a timestamp and `False` otherwise."""
        return isinstance(self._time, pd.Timestamp)

    def max(self) -> Union[Timedelta, Timestamp]:
        """Get the maximal time of the data."""
        if isinstance(self._time, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            return self._time.max()
        return self._time

    def min(self) -> Union[Timedelta, Timestamp]:
        """Get the minimal time of the data."""
        if isinstance(self._time, (pd.TimedeltaIndex, pd.DatetimeIndex)):
            return self._time.min()
        return self._time

    @staticmethod
    def _convert_quantity(
        time: pint.Quantity,
    ) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
        """Build a time-like pandas.Index from pint.Quantity."""
        time_ref = getattr(time, "time_ref", None)
        base = "s"  # using low base unit could cause rounding errors
        if not np.iterable(time):  # catch zero-dim arrays
            time = np.expand_dims(time, 0)
        delta = pd.TimedeltaIndex(data=time.to(base).magnitude, unit=base)
        if time_ref is not None:
            delta = delta + time_ref
        return delta

    @staticmethod
    def _convert_xarray(
        time: Union[xr.DataArray, xr.Dataset]
    ) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
        """Build a time-like pandas.Index from xarray objects."""
        if "time" in time.coords:
            time = time.time
        time_ref = time.weldx.time_ref
        time_index = pd.Index(time.values)
        if time_ref is not None:
            time_index = time_index + time_ref
        return time_index

    @staticmethod
    def _convert_other(time) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
        """Try autocasting input to time-like pandas index."""
        _input_type = type(time)

        if (not np.iterable(time) or isinstance(time, str)) and not isinstance(
            time, np.ndarray
        ):
            time = [time]

        time = pd.Index(time)

        if isinstance(time, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            return time
        # try manual casting for object dtypes (i.e. strings), should avoid integers
        # warning: this allows something like ["1","2","3"] which will be ns !!
        if is_object_dtype(time):
            for func in (pd.DatetimeIndex, pd.TimedeltaIndex):
                try:
                    return func(time)
                except (ValueError, TypeError):
                    continue

        raise TypeError(
            f"Could not convert {_input_type} "
            f"to pd.DatetimeIndex or pd.TimedeltaIndex"
        )

    @staticmethod
    def union(times=List[Union[types_time_like, "Time"]]) -> Time:
        """Calculate the union of multiple `Time` instances (or supported objects).

        Any reference time information will be dropped.

        Parameters
        ----------
        times
            A list of time class instances

        Returns
        -------
        weldx.Time
            The time union

        """
        pandas_index = reduce(
            lambda x, y: x.union(y),
            (Time(time).as_pandas_index() for time in times),
        )
        return Time(pandas_index)
