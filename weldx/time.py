"""Contains classes and functions related to time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp
from pandas.api.types import is_object_dtype
from xarray import DataArray

from weldx.constants import Q_

__all__ = [
    "Time",
    "TimeDependent",
    "types_timestamp_like",
    "types_datetime_like",
    "types_timedelta_like",
    "types_pandas_times",
    "types_time_like",
]


class TimeDependent(ABC):
    """An abstract base class that describes a common interface of time dep. classes."""

    @property
    @abstractmethod
    def time(self) -> Time:
        """Get the classes time component."""

    @property
    @abstractmethod
    def reference_time(self) -> Timestamp | None:
        """Return the reference timestamp if the time data is absolute."""


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
    from many data types and offers methods to convert back to one of the
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

    The underlying implementation is based on the core `pandas.TimedeltaIndex` and
    `pandas.DatetimeIndex` types, see the documentation for references.

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

    Raises
    ------
    ValueError
        When time values passed are not sorted in monotonic increasing order.

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

    Types that are derived from the abstract base class ``TimeDependent`` can also be
    passed directly to `Time` as `time` parameter:

    >>> from weldx import LocalCoordinateSystem as LCS
    >>> lcs = LCS(coordinates=Q_([[0, 0, 0], [1, 1, 1]], "mm"), time=["1s", "2s"])
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

    Subtraction is possible too, but there are some restrictions. It is not possible
    to subtract an absolute time from a time delta. Additionally, since the values of
    a ``Time`` instance must be monotonically increasing, any subtraction that
    produces a result that doesn't fulfill this requirement will fail. This is always
    the case when subtracting arrays from scalars because either the array that is
    subtracted (it is internally cast to a `Time` instance) or the resulting array
    violates this requirement.  Apart from that, subtraction works pretty similar as
    the addition:

    >>> # absolute and time delta
    >>> t_res = Time("2002") - "1d"
    >>> t_res = Time(["2002", "2007", "2022"]) - "1d"
    >>> t_res = Time(["2002", "2007", "2022"]) - ["1d", "2d", "3d"]
    >>>
    >>> # both absolute
    >>> t_res = Time(["2002"]) - "2001"
    >>> t_res = Time(["2002", "2007", "2022"]) - "2001"
    >>> t_res = Time(["2002", "2007", "2022"]) - ["2001", "2002", "2003"]
    >>>
    >>> # both time delta
    >>> t_res = Time("2d") - "1d"
    >>> t_res = Time(["2d", "7d", "22d"]) - "1d"
    >>> t_res = Time(["2d", "7d", "22d"]) - ["1d", "2d", "3d"]

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

    If you want to know how many entries are stored in a ``Time`` object, you can either
    use the ``length`` property or use ``len``:

    >>> len(Time(["1s", "3s"]))
    2

    Direct access and iteration are also supported. The return types are fitting pandas
    types:


    >>> t = Time(["1s", "2s", "3s"])
    >>> t[1]
    Time:
    0 days 00:00:02

    >>> t = Time(["2000", "2001", "2002"])
    >>> t[1]
    Time:
    2001-01-01 00:00:00
    reference time: 2000-01-01 00:00:00

    >>> from pandas import Timedelta
    >>>
    >>> t = Time(["1s", "2s", "3s"])
    >>> result = Timedelta(0, "s")
    >>>
    >>> for value in t:
    ...     if not isinstance(value, Timedelta):
    ...         raise TypeError("Unexpected type")
    ...     result += value
    >>>
    >>> result
    Timedelta('0 days 00:00:06')

    """

    def __init__(
        self,
        time: types_time_like,
        time_ref: types_timestamp_like = None,
    ):
        time, time_ref = self._get_time_input(time, time_ref)

        # sanity check
        if not isinstance(time, _data_base_types):
            raise TypeError("Could not create pandas time-like object.")

        if time_ref is not None:
            time_ref = pd.Timestamp(time_ref)
            if isinstance(time, pd.Timedelta):
                time = time + time_ref

        if isinstance(time, pd.TimedeltaIndex) and (time_ref is not None):
            time = time + time_ref

        if isinstance(time, pd.Index) and not time.is_monotonic_increasing:
            raise ValueError("The time values passed are not monotonic increasing.")

        self._time: pd.TimedeltaIndex | pd.DatetimeIndex = time
        self._time_ref: pd.Timestamp = time_ref

    @staticmethod
    def _get_time_input(time, time_ref):
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
        return time, time_ref

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

    def __eq__(self, other: types_time_like) -> bool | list[bool]:  # type: ignore
        """Element-wise comparisons between time object and compatible types.

        See Also
        --------
        equals : Check equality of `Time` objects.
        """
        return self._time == Time(other).as_pandas()

    __hash__ = None

    def __len__(self):
        """Return the length of the data."""
        return self.as_pandas_index().__len__()

    def __iter__(self):
        """Use generator to iterate over index values."""
        return (t for t in self.as_pandas_index())

    def __getitem__(self, item) -> Time:
        """Access pandas index."""
        return Time(self.as_pandas_index()[item], self.reference_time)

    def __getattr__(self, item: str):
        """Delegate unknown method calls to pandas index.

        Raises
        ------
        AttributeError
            When accessing a not implemented 'dunder' method or the requested method
            can not be accessed on the pandas index type.

        """
        if item.startswith("__"):
            raise AttributeError(f"Dunder method '{item}' not implemented for 'Time'.")
        try:
            return getattr(self.as_pandas_index(), item)
        except AttributeError as ae:
            raise AttributeError(
                f"Neither 'Time' object nor its pandas index has attribute '{item}'"
            ) from ae

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
        other = Time(other)
        if self.reference_time != other.reference_time:
            return False
        return np.allclose(self.as_quantity("s").m, other.as_quantity("s").m)

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
            # store time_ref info
            q.time_ref = self.reference_time  # type: ignore[attr-defined]
        return q  # type: ignore[return-value]

    def as_timedelta(self) -> Timedelta | TimedeltaIndex:
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

    def as_datetime(self) -> Timestamp | DatetimeIndex:
        """Return the data as `pandas.DatetimeIndex`."""
        if not self.is_absolute:
            raise TypeError("Cannot convert non absolute Time object to datetime")
        return self._time

    def as_pandas(
        self,
    ) -> pd.Timedelta | pd.TimedeltaIndex | pd.Timestamp | pd.DatetimeIndex:
        """Return the underlying pandas time datatype."""
        return self._time

    def as_pandas_index(self) -> pd.TimedeltaIndex | pd.DatetimeIndex:
        """Return a pandas index type regardless of length.

        This is useful when using time as coordinate in xarray types.
        """
        if isinstance(self._time, pd.Timestamp):
            return pd.DatetimeIndex([self._time])
        if isinstance(self._time, pd.Timedelta):
            return pd.TimedeltaIndex([self._time])
        return self._time

    def as_data_array(self, timedelta_base: bool = True) -> DataArray:
        """Return the time data as a `xarray.DataArray` coordinate.

        By default the format is timedelta values with reference time as attribute.

        Parameters
        ----------
        timedelta_base
            If true (the default) the values of the xarray will always be timedeltas.

        """
        if timedelta_base:
            t = self.as_timedelta_index()
        else:
            t = self.index
        da = xr.DataArray(t, coords={"time": t}, dims=["time"])
        if self.reference_time is not None:
            da.weldx.time_ref = self.reference_time

        da.attrs = da.time.attrs
        return da

    @property
    def reference_time(self) -> Timestamp | None:
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
    def is_timestamp(self) -> bool:
        """Return `True` if the data represents a timestamp and `False` otherwise."""
        return isinstance(self._time, pd.Timestamp)

    @property
    def index(self) -> pd.TimedeltaIndex | pd.DatetimeIndex:
        """Return a pandas index type regardless of length.

        See Also
        --------
        `Time.as_pandas_index`

        """
        return self.as_pandas_index()

    @property
    def timedelta(self) -> pd.TimedeltaIndex:
        """Return the timedelta values relative to the reference time (if it exists).

        See Also
        --------
        `Time.as_timedelta_index`

        """
        return self.as_timedelta_index()

    @property
    def quantity(self) -> pint.Quantity:
        """Return the `pint.Quantity` representation scaled to seconds.

        See Also
        --------
        `Time.as_quantity`

        """
        return self.as_quantity(unit="s")  # type: ignore[return-value]

    @property
    def duration(self) -> Time:
        """Get the covered time span."""
        return Time(self.max() - self.min())

    def resample(self, number_or_interval: int | types_timedelta_like):
        """Resample the covered duration.

        Parameters
        ----------
        number_or_interval :
            If an integer is passed, the covered time period will be divided into
            equally sized time steps so that the total number of time steps is equal to
            the passed number. If a timedelta is passed, the whole period will be
            resampled so that the difference between all time steps matches the
            timedelta. Note that the boundaries of the time period will not change.
            Therefore, the timedelta between the last two time values might differ from
            the desired timedelta.

        Returns
        -------
        weldx.time.Time :
            Resampled time object

        Raises
        ------
        RuntimeError
            When the time data consists only of a single value and has no duration.
        TypeError
            When the passed value is neither an integer or a supported time delta value
        ValueError
            When the passed time delta is equal or lower than 0

        Examples
        --------
        >>> from weldx import Time
        >>> t = Time(["3s","6s","7s", "9s"])

        Resample using an integer:

        >>> t.resample(4)
        Time:
        TimedeltaIndex(['0 days 00:00:03', '0 days 00:00:05', '0 days 00:00:07',
                        '0 days 00:00:09'],
                       dtype='timedelta64[ns]', freq=None)

        Resample with a time delta:

        >>> t.resample("1.5s")
        Time:
        TimedeltaIndex([       '0 days 00:00:03', '0 days 00:00:04.500000',
                               '0 days 00:00:06', '0 days 00:00:07.500000',
                               '0 days 00:00:09'],
                       dtype='timedelta64[ns]', freq='1500ms')

        """
        if len(self) <= 1:
            raise RuntimeError("Can't resample a single time delta or timestamp")

        tdi = self.as_timedelta_index()
        t0, t1 = tdi.min(), tdi.max()

        if isinstance(number_or_interval, int):
            if number_or_interval < 2:
                raise ValueError("Number of time steps must be equal or larger than 2.")

            tdi_new = pd.timedelta_range(start=t0, end=t1, periods=number_or_interval)
        else:
            freq = Time(number_or_interval).as_timedelta()

            if freq <= pd.Timedelta(0):
                raise ValueError("Time delta must be a positive, non-zero value.")

            tdi_new = pd.timedelta_range(start=t0, end=t1, freq=freq)

        if not tdi_new[-1] == t1:
            tdi_new = tdi_new.append(pd.Index([t1]))

        return Time(tdi_new, self.reference_time)

    @staticmethod
    def _convert_quantity(
        time: pint.Quantity,
    ) -> pd.TimedeltaIndex | pd.DatetimeIndex:
        """Build a time-like pandas.Index from pint.Quantity."""
        time_ref = getattr(time, "time_ref", None)
        base = "s"  # using low base unit could cause rounding errors

        if not np.iterable(time):  # catch zero-dim arrays
            # The mypy error in the next line is ignored. `np.expand_dims` only expects
            # `ndarray` types and does not know about quantities, but pint provides the
            # necessary interfaces so that the function works as expected
            time = np.expand_dims(time, 0)  # type: ignore[assignment]

        delta = pd.to_timedelta(time.to(base).magnitude, base)
        if time_ref is not None:
            delta = delta + time_ref
        return delta

    @staticmethod
    def _convert_xarray(
        time: xr.DataArray | xr.Dataset,
    ) -> pd.TimedeltaIndex | pd.DatetimeIndex:
        """Build a time-like pandas.Index from xarray objects."""
        if "time" in time.coords:
            time = time.time
        time_ref = time.weldx.time_ref
        if time.shape:
            time_index = pd.Index(time.values)
        else:
            time_index = pd.Index([time.values])
        if time_ref is not None:
            time_index = time_index + time_ref
        return time_index

    @staticmethod
    def _convert_other(time) -> pd.TimedeltaIndex | pd.DatetimeIndex:
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
            f"Could not convert {_input_type} to pd.DatetimeIndex or pd.TimedeltaIndex"
        )

    class _UnionDescriptor:
        """Enables different behavior of `.union` as class and instance method."""

        def __get__(self, ins, typ):
            if ins is None:
                return typ._union_class
            return ins._union_instance

    # todo: Sphinx renders the docstring below as expected. However, the examples are
    #       not run by doctest. A possible solution in Python 3.9 can be found here:
    #       https://stackoverflow.com/a/8820636/6700329
    union = _UnionDescriptor()
    """Calculate the union of multiple time-like objects.

    This method can either be used as a class or instance method. When used on an
    instance, its values are included in the calculated time union.

    Note that any reference time information will be dropped.

    Parameters
    ----------
    times:
        A list of time-like objects

    Returns
    -------
    Time
        The time union

    Examples
    --------

    Using ``union`` as class method:

    >>> from weldx import Time
    >>> t1 = Time(["1s", "3s", "4s"])
    >>> t2 = Time(["2s", "4s", "5s"])
    >>>
    >>> all(Time.union([t1, t2]) == Time(["1s", "2s", "3s", "4s", "5s"]))
    True

    Using the instance method:

    >>> all(t1.union([t2]) == Time(["1s", "2s", "3s", "4s", "5s"]))
    True

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


# list of types that are supported to be stored in Time._time
_data_base_types = (pd.Timedelta, pd.Timestamp, pd.DatetimeIndex, pd.TimedeltaIndex)

types_datetime_like = Union[DatetimeIndex, np.datetime64, list[str], Time]
"""types that define ascending arrays of time stamps."""

types_timestamp_like = Union[Timestamp, str, Time]
"""types that define timestamps."""

types_timedelta_like = Union[
    TimedeltaIndex, pint.Quantity, np.timedelta64, list[str], Time
]
"""types that define ascending time delta arrays."""

types_time_like = Union[
    types_datetime_like, types_timedelta_like, types_timestamp_like, TimeDependent
]
"""types that represent time."""

types_pandas_times = Union[Timedelta, Timestamp, DatetimeIndex, TimedeltaIndex]
"""supported pandas time types."""
