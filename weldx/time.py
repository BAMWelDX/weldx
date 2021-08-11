"""Contains classes and functions related to time."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp
from pandas.api.types import is_object_dtype
from xarray import DataArray

from .constants import Q_

__all__ = [
    "Time",
    "TimeDependent",
    "types_timestamp_like",
    "types_datetime_like",
    "types_timedelta_like",
    "types_pandas_times",
    "types_time_like",
]


class Time:
    """Provides a unified interface for time related operations.

    The purpose of this class is to provide a unified interface for all operations
    related to time. This is important because time can have multiple representations.
    When working with time, some difficulties that might arise are the following:

        - we can have absolute times in form of dates and relative times in form of
          quantities
        - conversion factors between different time quantities differ. An hour consists
          of 60 minutes, but a day has only 24 hours
        - there are multiple time data types available in python like the ones provided
          by numpy or pandas. If you have to work with time classes from multiple
          libraries you might have to do a lot of conversions to perform simple tasks as
          calculating a time delta between two dates.

    This class solves the mentioned problems for many cases. It can be created
    from many different data types and offers methods to convert back to one of the
    supported types. Most of its methods also support the date types that can be used to
    create an instance of this class. Therefore, you do not need to perform any
    conversions yourself.

    You can create the class from the following time representations:

        - other instances of the ``Time`` class
        - numpy: ``datetime64`` and ``timedelta64``
        - pandas: ``Timedelta``, ``Timestamp``, ``TimedeltaIndex``, ``DatetimeIndex``
        - `pint.Quantity`
        - strings representing a date (``"2001-01-23 14:23:11"``) or a timedelta
          (``23s``)

    Parameters
    ----------
    time :
        A supported class that represents either absolute or relative times. The
        data must be in ascending order. All classes derived from the abstract base
        class 'TimeDependent' are supported too.
    time_ref :
        An absolute reference point in time (timestamp). The return values of all
        accessors that return relative times will be calculated towards this
        reference time.
        This will turn the data of this class into absolute time values if relative
        times are passed as ``time`` parameter. In case ``time`` already contains
        absolute values and this parameter is set to ``None``, the first value of
        the data will be used as reference time.

    Examples
    --------
    Creation from a quantity:

    >>> from weldx import Q_, Time
    >>>
    >>> quantity = Q_("10s")
    >>> t_rel = Time(quantity)

    Since a quantity is not an absolute time like a date, the ``is_absolute`` property
    is ``False``:

    >>> t_rel.is_absolute
    False

    To create an absolute value, just add a time stamp as ``time_ref`` parameter:

    >>> from pandas import Timestamp
    >>>
    >>> timestamp = Timestamp("2042-01-01 13:37")
    >>> t_abs = Time(quantity, timestamp)
    >>> t_abs.is_absolute
    True

    Or use an absolute time type:

    >>> t_abs = Time(timestamp)
    >>> t_abs.is_absolute
    True

    >>> from pandas import DatetimeIndex
    >>>
    >>> dti = DatetimeIndex(["2001", "2002"])
    >>> t_abs = Time(dti)
    >>> t_abs.is_absolute
    True

    If you want to create a ``Time`` instance without importing anything else, just use
    strings:

    >>> # relative times
    >>> t_rel = Time("1h")
    >>> t_rel = Time(["3s","3h","3d"])
    >>>
    >>> # absolute times
    >>> t_abs = Time(["1s","2s","3s"],"2010-10-05 12:00:00")
    >>> t_abs = Time("3h", "2010-08-11")
    >>> t_abs = Time("2014-07-23")
    >>> t_abs = Time(["2000","2001","2002"])

    Types that are derived from the abstract base class `TimeDependent` can also be
    passed directly to `Time` as `time` parameter:

    >>> from weldx import LocalCoordinateSystem as LCS
    >>> lcs = LCS(coordinates=[[0, 0, 0], [1, 1, 1]], time=["1s", "2s"])
    >>> t_from_lcs = Time(lcs)
    >>>
    >>> from weldx import TimeSeries
    >>> ts = TimeSeries(Q_([4, 5, 6], "m"), ["2000", "2001", "2002"])
    >>> t_from_ts = Time(ts)

    As long as one of the operands represents a timedelta, you can add two `Time`
    instances. If one of the instances is an array, the other one needs to be either a
    scalar or an array of same length. In the latter case, values are added per index:

    >>> t_res = Time(["1s", "2s"]) + Time("3s")
    >>> t_res = Time(["1s", "2s"]) + Time(["3s", "4s"])
    >>>
    >>> t_res = Time(["1d", "2d"]) + Time("2000-01-01")
    >>> t_res = Time(["2001-01-01", "2001-01-02"]) + Time(["3d", "4d"])

    `Time` also accepts all other supported types on the right hand side of the ``+``
    operator:

    >>> t_res = Time(["1s", "2s"]) + Q_("10s")
    >>> t_res = Time(["1s", "2s"]) + DatetimeIndex(["2001", "2002"])
    >>> t_res = Time(["1d", "2d"]) + "2000-01-01"
    >>> t_res = Time(["1s", "2s"]) + ["3s", "4s"]

    Except for the numpy types and `pint.Quantity` other types are also supported on
    the left hand side:

    >>> t_res = DatetimeIndex(["2001", "2002"]) + Time(["1d", "2d"])
    >>> t_res = "2000-01-01" + Time(["1d", "2d"])
    >>> t_res = ["3s", "4s"] + Time(["1s", "2s"])

    You can also compare two instances of `Time`:

    >>> Time(["1s"]) == Time(Q_("1s"))
    True

    >>> Time("1s") == Time("2s")
    False

    >>> Time("2000-01-01 17:00:00") == Time("2s")
    False

    Note that any provided reference time is not taken into account when comparing two
    absolute time values. Only the values itself are compared:

    >>> dti = DatetimeIndex(["2001", "2002", "2003"])
    >>> all(Time(dti, "2000") == Time(dti, "2042"))
    True

    If you want to include the reference times into the comparison, use the `equals`
    method.

    All supported types can also be used on the right hand side of the ``==`` operator:

    >>> all(Time(["2000", "2001"]) == DatetimeIndex(["2000", "2001"]))
    True

    >>> all(Time(["1s", "2s"]) == Q_([1, 2],"s"))
    True

    >>> Time("3s") == "20d"
    False

    Raises
    ------
    ValueError
        When time values passed are not sorted in monotonic increasing order.

    """

    def __init__(
        self,
        time: types_time_like,
        time_ref: types_timestamp_like = None,
    ):
        # todo: update type hints (see: https://stackoverflow.com/q/46092104/6700329)
        # problem: ring dependency needs to be solved
        if issubclass(type(time), TimeDependent):
            time = time.time  # type: ignore[union-attr] # mypy doesn't filter correctly
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

        self._time: Union[pd.TimedeltaIndex, pd.DatetimeIndex] = time
        self._time_ref: pd.Timestamp = time_ref

    def __add__(self, other: types_time_like) -> Time:
        """Element-wise addition between `Time` object and compatible types."""
        other = Time(other)
        time_ref = self.reference_time if self.is_absolute else other.reference_time
        return Time(self._time + other.as_pandas(), time_ref)

    def __radd__(self, other: types_time_like) -> Time:
        """Element-wise addition between `Time` object and compatible types."""
        return self + other

    def __sub__(self, other: types_time_like) -> Time:
        """Element-wise subtraction between `Time` object and compatible types."""
        other = Time(other)
        time_ref = None if other.is_absolute else self.reference_time
        return Time(self._time - other.as_pandas(), time_ref)

    def __rsub__(self, other: types_time_like) -> Time:
        """Element-wise subtraction between `Time` object and compatible types."""
        other = Time(other)
        time_ref = None if self.is_absolute else other.reference_time
        return Time(other.as_pandas() - self._time, time_ref)

    def __eq__(self, other: types_time_like) -> Union[bool, List[bool]]:  # type: ignore
        """Element-wise comparisons between time object and compatible types.

        See Also
        --------
        equals : Check equality of `Time` objects.
        """
        return self._time == Time(other).as_pandas()

    def __len__(self):
        """Return the length of the data."""
        return self.length

    def __iter__(self):
        """Use generator to iterate over index values."""
        return (t for t in self.as_pandas_index())

    def __getitem__(self, item):
        """Access pandas index."""
        return self.as_pandas_index()[item]

    def __repr__(self):
        """Console info."""
        repr_str = "Time:\n" + self.as_pandas().__str__()
        if self.is_absolute:
            repr_str = repr_str + f"\nreference time: {str(self.reference_time)}"
        return repr_str

    def equals(self, other: Time) -> bool:
        """Test for matching ``time`` and ``reference_time`` between objects.

        Parameters
        ----------
        other :
            The compared object

        Returns
        -------
        bool :
            `True` if both objects have the same time values and reference time, `False`
            otherwise

        """
        return np.all(self == other) & (self._time_ref == other._time_ref)

    def all_close(self, other: types_time_like) -> bool:
        """Return `True` if another object compares equal within a certain tolerance.

        Parameters
        ----------
        other :
            The compared object

        Returns
        -------
        bool :
            `True` if all time values compare equal within a certain tolerance, `False`
            otherwise

        """
        # TODO: handle tolerances ?
        return np.allclose(self._time, Time(other).as_pandas())

    def as_quantity(self, unit: str = "s") -> pint.Quantity:
        """Return the data as `pint.Quantity`.

        Parameters
        ----------
        unit :
            String that specifies the desired time unit for conversion.

        Returns
        -------
        pint.Quantity :
            Converted time quantity

        Notes
        -----
        from pandas Timedelta documentation: "The .value attribute is always in ns."
        https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
        """
        nanoseconds = self.as_timedelta_index().values.astype(np.int64)
        if len(nanoseconds) == 1:
            nanoseconds = nanoseconds[0]
        q = Q_(nanoseconds, "ns").to(unit)
        if self.is_absolute:
            setattr(q, "time_ref", self.reference_time)  # store time_ref info
        return q

    def as_timedelta(self) -> Union[Timedelta, TimedeltaIndex]:
        """Return the data as `pandas.TimedeltaIndex` or `pandas.Timedelta`."""
        if self.is_absolute:
            return self._time - self.reference_time
        return self._time

    def as_timedelta_index(self) -> TimedeltaIndex:
        """Return the data as `pandas.TimedeltaIndex`."""
        timedelta = self.as_timedelta()
        if isinstance(timedelta, Timedelta):
            return TimedeltaIndex([timedelta])
        return timedelta

    def as_timestamp(self) -> Timestamp:
        """Return a `pandas.Timestamp` if the object represents a timestamp."""
        if not isinstance(self._time, Timestamp):
            raise TypeError("Time object does not represent a timestamp.")
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
    def reference_time(self, time_ref: types_timestamp_like):
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

    class _UnionDescriptor:
        """Enables different behavior of `.union` as class and instance method."""

        def __get__(self, ins, typ):
            if ins is None:
                return typ._union_class
            return ins._union_instance

    union = _UnionDescriptor()
    """Calculate the union of multiple time-like objects.

    This method can eiter be used as a class or instance method. When used on an 
    instance, its values are included in the calculated time union.
    
    Note that any reference time information will be dropped.

    Parameters
    ----------
    times: Sequence[types_time_like]
        A list of time-like objects

    Returns
    -------
    Time
        The time union

    """

    @staticmethod
    def _union_class(times: Sequence[types_time_like]) -> Time:
        """Class version of the ``union`` method."""
        pandas_index = reduce(
            lambda x, y: x.union(y),
            (Time(time).as_pandas_index() for time in times),
        )
        return Time(pandas_index)

    def _union_instance(self, times: Sequence[types_time_like]) -> Time:
        """Instance version of the ``union`` method."""
        return Time._union_class([self, *times])


class TimeDependent(ABC):
    """An abstract base class that describes a common interface of time dep. classes."""

    @property
    @abstractmethod
    def time(self) -> Time:
        """Get the classes time component."""

    @property
    @abstractmethod
    def reference_time(self) -> Union[Timestamp, None]:
        """Return the reference timestamp if the time data is absolute."""


# list of types that are supported to be stored in Time._time
_data_base_types = (pd.Timedelta, pd.Timestamp, pd.DatetimeIndex, pd.TimedeltaIndex)

types_datetime_like = Union[DatetimeIndex, np.datetime64, List[str], Time]
"""types that define ascending arrays of time stamps."""

types_timestamp_like = Union[Timestamp, str, Time]
"""types that define timestamps."""

types_timedelta_like = Union[
    TimedeltaIndex, pint.Quantity, np.timedelta64, List[str], Time
]
"""types that define ascending time delta arrays."""

types_time_like = Union[
    types_datetime_like, types_timedelta_like, types_timestamp_like, TimeDependent
]
"""types that represent time."""

types_pandas_times = Union[Timedelta, Timestamp, DatetimeIndex, TimedeltaIndex]
"""supported pandas time types."""
