"""Common type definitions"""
import pathlib
from io import IOBase
from typing import runtime_checkable, Protocol, Union


@runtime_checkable
class SupportsFileReadOnly(Protocol):
    """Type interface for read()."""

    __slots__ = ()

    def read(self):
        raise NotImplementedError


@runtime_checkable
class SupportsFileReadWrite(Protocol):
    """Type interface for read, write and seeking."""

    __slots__ = ()

    def read(self):
        raise NotImplementedError

    def write(self, *args):
        raise NotImplementedError

    def tell(self):
        raise NotImplementedError

    def seek(self, *args):
        raise NotImplementedError


types_file_like = Union[IOBase, SupportsFileReadOnly, SupportsFileReadWrite]
types_path_and_file_like = Union[str, pathlib.Path, types_file_like]
