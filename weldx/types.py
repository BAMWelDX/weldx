"""Common type definitions."""
import pathlib
from io import IOBase
from typing import Protocol, Union, runtime_checkable

from pandas import DatetimeIndex, TimedeltaIndex, Timestamp
from pint import Quantity

__all__ = [
    "SupportsFileReadOnly",
    "SupportsFileReadWrite",
    "types_file_like",
    "types_path_like",
    "types_path_and_file_like",
]


@runtime_checkable
class SupportsFileReadOnly(Protocol):
    """Type interface for read()."""

    __slots__ = ()

    def read(self):
        """Read content."""

    def readline(self):
        """Read a line."""


@runtime_checkable
class SupportsFileReadWrite(Protocol):
    """Type interface for read, write and seeking."""

    __slots__ = ()

    def read(self):
        """Read content."""

    def readline(self):
        """Read a line."""

    def write(self, *args):
        """Write content."""

    def tell(self):
        """Get position."""

    def seek(self, *args):
        """Go to position."""


types_file_like = Union[IOBase, SupportsFileReadOnly, SupportsFileReadWrite]
"""types which support reading, writing, and seeking."""

types_path_like = Union[str, pathlib.Path]
"""types defining a path to a file/directory and can be passed to `open`."""

types_path_and_file_like = Union[types_path_like, types_file_like]
"""types to handle paths and file handles."""

types_datetime_like = Union[DatetimeIndex]
"""types that define ascending arrays of time stamps."""

types_timestamp_like = Union[Timestamp, str]
"""types that define timestamps."""

types_timedelta_like = Union[TimedeltaIndex, Quantity]
"""types that define ascending time delta arrays."""

types_time_like = Union[types_datetime_like, types_timedelta_like, types_timestamp_like]
"""types that represent time."""
