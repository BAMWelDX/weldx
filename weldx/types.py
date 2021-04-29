"""Common type definitions."""
import pathlib
from io import IOBase
from typing import Protocol, Union, runtime_checkable


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
types_path_and_file_like = Union[str, pathlib.Path, types_file_like]
