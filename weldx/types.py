"""Common type definitions."""
from io import IOBase
from pathlib import Path
from typing import Protocol, Union, runtime_checkable

__all__ = [
    "SupportsFileReadOnly",
    "SupportsFileReadWrite",
    "types_file_like",
    "types_path_and_file_like",
]


@runtime_checkable
class SupportsFileReadOnly(Protocol):
    """Type interface for read()."""

    __slots__ = ()

    def read(self):
        """Read content."""


@runtime_checkable
class SupportsFileReadWrite(Protocol):
    """Type interface for read, write and seeking."""

    __slots__ = ()

    def read(self):
        """Read content."""

    def write(self, *args):
        """Write content."""

    def tell(self):
        """Get position."""

    def seek(self, *args):
        """Go to position."""


types_file_like = Union[IOBase, SupportsFileReadOnly, SupportsFileReadWrite]
"""types which support reading, writing, and seeking."""

types_path_and_file_like = Union[str, Path, types_file_like]
"""types which can be passed to `open`"""
