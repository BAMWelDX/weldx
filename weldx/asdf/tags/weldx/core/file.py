"""Contains classes for the asdf serialization of an external file."""


import socket
from copy import deepcopy
from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from weldx.asdf.types import WeldxType

# Python class -------------------------------------------------------------------------


@dataclass
class ExternalFile:
    """Handles the asdf serialization of external files."""

    path: Union[str, Path] = None

    filename: str = None
    suffix: str = None
    directory: str = None
    hostname: str = None

    created: pd.Timestamp = None
    modified: pd.Timestamp = None
    size: int = None

    hashing_algorithm: str = "SHA-256"
    hash: str = None
    asdf_save_content: bool = False
    buffer: bytes = None

    hash_mapping = {"MD5": md5, "SHA-256": sha256}

    def __post_init__(self):
        """Initialize the internal values."""
        if self.path is not None:
            if not isinstance(self.path, Path):
                self.path = Path(self.path)
            if not self.path.is_file():
                raise ValueError(f"File not found: {self.path.as_posix()}")

            self.filename = self.path.name
            self.suffix = "".join(self.path.suffixes)[1:]
            self.directory = self.path.parent.absolute().as_posix()
            if self.hostname is None:
                self.hostname = socket.gethostname()

            stat = self.path.stat()
            self.size = stat.st_size
            self.created = pd.Timestamp(stat.st_ctime_ns)
            self.modified = pd.Timestamp(stat.st_mtime_ns)

            self.hashing_algorithm = self.hashing_algorithm.upper()
            if self.hashing_algorithm not in ExternalFile.hash_mapping:
                raise ValueError(
                    f"'{self.hashing_algorithm}' is not a supported hashing algorithm."
                )

    @staticmethod
    def calculate_hash(
        path_or_buffer: Union[str, Path, bytes],
        algorithm: str,
        buffer_size: int = 65536,
    ) -> str:
        """Calculate the hash of a file.

        Parameters
        ----------
        path_or_buffer : Union[str, pathlib.Path, bytes]
            Path of the file or buffer as bytes
        algorithm : str
            Name of the desired hashing algorithm
        buffer_size : int
            Size of the internally used buffer. The file will be read in
            corresponding chunks. No effect when hashing from buffer.

        Returns
        -------
        str :
            The calculated hash

        """
        hashing_class = ExternalFile.hash_mapping[algorithm.upper()]()
        if isinstance(path_or_buffer, bytes):
            hashing_class.update(path_or_buffer)
        else:
            with open(path_or_buffer, "rb") as file:
                while True:
                    data = file.read(buffer_size)
                    if not data:
                        break
                    hashing_class.update(data)
        return hashing_class.hexdigest()

    def get_file_content(self) -> bytes:
        """Get the contained bytes of the file.

        Returns
        -------
        bytes :
            The file's content

        """
        return self.path.read_bytes()

    def write_to(self, directory: Union[str, Path], file_system=None):
        """Write the file to the specified destination.

        Parameters
        ----------
        directory : Union[str, pathlib.Path]
            Directory where the file should be written.
        file_system :
            The target file system.

        """
        path = Path(f"{directory}/{self.filename}")

        buffer = self.buffer
        if buffer is None:
            buffer = self.get_file_content()

        if file_system is None:
            path.write_bytes(buffer)
        else:
            file_system.writebytes(path.as_posix(), buffer)


# ASDF Serialization -------------------------------------------------------------------


class FileTypeASDF(WeldxType):
    """Serialization class for `weldx.core.ExternalFile`."""

    name = "core/file"
    version = "1.0.0"
    types = [ExternalFile]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: ExternalFile, ctx):
        """
        Convert an 'weldx.core.ExternalFile' instance into YAML  representations.

        Parameters
        ----------
        node :
            Instance of the 'weldx.core.ExternalFile' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the
            'weldx.core.ExternalFile' type to be serialized.

        """
        tree = deepcopy(node.__dict__)

        path = tree.pop("path", None)
        buffer = tree.pop("buffer", None)
        save_content = tree.pop("asdf_save_content")
        algorithm = tree.pop("hashing_algorithm")
        hash_value = tree.pop("hash")

        if save_content:
            if buffer is None:
                buffer = node.get_file_content()
            tree["content"] = np.frombuffer(buffer, dtype=np.uint8)

        if buffer is None:
            if hash_value is None and path is not None:
                hash_value = node.calculate_hash(path, node.hashing_algorithm)
        else:
            hash_value = node.calculate_hash(buffer, node.hashing_algorithm)

        if hash_value is not None:
            tree["content_hash"] = {"algorithm": algorithm, "value": hash_value}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into an
        'weldx.core.ExternalFile'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        weldx.core.ExternalFile :
            An instance of the 'weldx.core.ExternalFile' type.

        """
        buffer = tree.pop("content", None)
        if buffer is not None:
            buffer = buffer.tobytes()
            tree["buffer"] = buffer

        hash_data = tree.pop("content_hash", None)
        if hash_data is not None:
            tree["hashing_algorithm"] = hash_data["algorithm"]
            tree["hash"] = hash_data["value"]

        if buffer is not None:
            hash_buffer = ExternalFile.calculate_hash(buffer, tree["hashing_algorithm"])
            if hash_buffer != tree["hash"]:  # pragma: no cover
                raise Exception(
                    "The stored hash does not match the stored contents' hash."
                )
        return ExternalFile(**tree)
