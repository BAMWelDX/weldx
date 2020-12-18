"""Contains the asdf serialization class for `weldx.core.ExternalFileBuffer`."""

from hashlib import sha256

import numpy as np

from weldx.asdf.types import WeldxType
from weldx.core import ExternalFile


class FileTypeASDF(WeldxType):
    """Serialization class for `weldx.core.ExternalFile`."""

    name = "core/file"
    version = "1.0.0"
    types = [ExternalFile]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def _get_hash(cls, buffer, algorithm: str = "SHA-256"):
        # https://www.freecodecamp.org/news/md5-vs-sha-1-vs-sha-2-which-is-the-most-secure-encryption-hash-and-how-to-check-them/
        # https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed
        if algorithm == "SHA-256":
            hasher = sha256()
            hasher.update(buffer)
            return hasher.hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

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
        hash_algorithm = "SHA-256"
        if node.asdf_save_content:
            buffer = node.get_file_content()
            buffer_hash = cls._get_hash(buffer, hash_algorithm)
            buffer_np = np.frombuffer(buffer, dtype=np.uint8)
            return {
                "filename": node.filename,
                "content": buffer_np,
                "content_hash": {"algorithm": hash_algorithm, "value": buffer_hash},
            }

        return {
            "filename": node.filename,
            "hostname": node.hostname,
            "location": node.location,
        }

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
        weldx.core.ExternalFileBuffer :
            An instance of the 'weldx.core.ExternalFile' type.

        """
        buffer = None
        hostname = None
        if "content" in tree:
            buffer = tree["content"].tobytes()
            hash_calc = cls._get_hash(buffer, tree["content_hash"]["algorithm"])
            hash_stored = tree["content_hash"]["value"]
            if not np.all(hash_calc == hash_stored):
                raise Exception("Invalid hash.")
            path = tree["filename"]
        else:
            path = f"{tree['location']}/{tree['filename']}"
            hostname = tree["hostname"]
        return ExternalFile(path, _buffer=buffer, hostname=hostname)
