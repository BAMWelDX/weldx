"""Common type definitions."""
import pathlib
from io import IOBase
from typing import Protocol, Union, runtime_checkable

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
