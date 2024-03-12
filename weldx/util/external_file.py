"""External file utilities."""

import mimetypes
import socket
from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from typing import Union

import pandas as pd

__all__ = ["ExternalFile"]


@dataclass
class ExternalFile:
    """Handles the asdf serialization of external files."""

    path: Union[Path] = None

    filename: str = None
    suffix: str = None
    mimetype: str = None
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
            self.mimetype = mimetypes.guess_type(self.path)[0]
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
