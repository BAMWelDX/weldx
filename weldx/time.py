"""Contains classes and functions related to time."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from pandas import DatetimeIndex, TimedeltaIndex, Timestamp
    from pint import Quantity
    from xarray import DataArray

    from weldx.types import types_time_like, types_timestamp_like


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

        """
        pass

    def __add__(self, other):
        # discuss what this is supposed to do. There are multiple possibilities
        pass

    def __radd__(self, other):
        # custom implementation for right hand syntax with other time-like types
        return self + other  # simply call normal __add__

    def __sub__(self, other):
        # discuss what this is supposed to do. There are multiple possibilities
        pass

    def __rsub__(self, other):
        # custom implementation for right hand syntax with other time-like types
        pass

    def __eq__(self, other: Union[types_time_like, Time]) -> bool:
        pass

    def __le__(self, other: Union[types_time_like, Time]) -> bool:
        pass

    def all_close(self, other: Union[types_time_like, Time], tolerance) -> bool:
        """Return `True` if another object compares equal within a certain tolerance."""

    def as_quantity(self) -> Quantity:
        """Return the data as `pint.Quantity`."""

    def as_time_delta_index(self) -> TimedeltaIndex:
        """Return the data as `pandas.TimedeltaIndex`."""

    def as_datetime_index(self) -> DatetimeIndex:
        """Return the data as `pandas.DatetimeIndex`."""

    def as_timestamp(self) -> Timestamp:
        """Return the data as `pandas.Timestamp`."""

    def as_pandas(self):
        pass
     

    @property
    def data_array(self) -> DataArray:
        """Return the internally stored data."""

    @data_array.setter
    def data_array(self, data_array: DataArray):
        """Set the internal data."""

    @property
    def reference_time(self) -> Time:
        """Get the reference time."""

    @property
    def reference_time_as_timestamp(self) -> Timestamp:
        """Get the reference time as `pandas.Timestamp`."""
        return self.reference_time.as_timestamp()

    @reference_time.setter
    def reference_time(self, time_ref: Union[types_timestamp_like, Time]):
        """Set the reference time."""
        pass

    @property
    def has_reference_time(self) -> bool:
        """Return `True` if the class has a reference time and `False` otherwise."""

    @property
    def length(self) -> int:
        """Return the length of the data."""

    @property
    def is_timestamp(self) -> bool:
        """Return `True` if the data represents a timestamp and `False` otherwise."""
        return self.length == 1 and self.has_reference_time

    def max(self) -> Time:
        """Get the maximal time of the data."""

    def min(self) -> Time:
        """Get the minimal time of the data."""

    @staticmethod
    def union(times=List[Union[types_time_like, "Time"]]) -> Time:
        """Calculate the union of multiple `Time` instances.

        Parameters
        ----------
        times :
            A list of time class instances

        Returns
        -------
        weldx.Time :
            The time union

        """
