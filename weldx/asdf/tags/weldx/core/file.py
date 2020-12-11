"""Contains the asdf serialization class for `weldx.core.ExternalFileBuffer`."""

from hashlib import sha256

import numpy as np

from weldx.asdf.types import WeldxType
from weldx.core import ExternalFileBuffer


class FileTypeASDF(WeldxType):
    """Serialization class for `weldx.core.ExternalFile`."""

    name = "core/file"
    version = "1.0.0"
    types = [ExternalFileBuffer]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def _get_hash(cls, buffer, algorithm: str = "SHA-256"):
        # https://www.freecodecamp.org/news/md5-vs-sha-1-vs-sha-2-which-is-the-most-secure-encryption-hash-and-how-to-check-them/
        # https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed
        if algorithm == "SHA-256":
            hasher = sha256()
            hasher.update(buffer)
            return np.frombuffer(hasher.digest(), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @classmethod
    def to_tree(cls, node: ExternalFileBuffer, ctx):
        """
        Convert an 'weldx.core.ExternalFileBuffer' instance into YAML  representations.

        Parameters
        ----------
        node :
            Instance of the 'weldx.core.ExternalFileBuffer' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the
            'weldx.core.ExternalFileBuffer' type to be serialized.

        """
        hash_algorithm = "SHA-256"
        buffer_hash = cls._get_hash(node.buffer, hash_algorithm)
        buffer_np = np.frombuffer(node.buffer, dtype=np.uint8)
        return {
            "filename": node.filename,
            "content": buffer_np,
            "content_hash": {"algorithm": hash_algorithm, "value": buffer_hash},
        }

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into an
        'weldx.core.ExternalFileBuffer'.

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
            An instance of the 'weldx.core.ExternalFileBuffer' type.

        """
        buffer = tree["content"].tobytes()
        hash_calc = cls._get_hash(buffer, tree["content_hash"]["algorithm"])
        hash_stored = tree["content_hash"]["value"]
        if not np.all(hash_calc == hash_stored):
            raise Exception("Invalid hash.")
        return ExternalFileBuffer(tree["filename"], buffer=buffer)
