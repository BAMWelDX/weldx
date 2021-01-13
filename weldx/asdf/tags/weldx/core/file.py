"""Contains the asdf serialization class for `weldx.core.ExternalFileBuffer`."""


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
            buffer_hash = node.calculate_hash(buffer, hash_algorithm)
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
            "size": node.size,
            "created": node.created,
            # "modified": node.modified,
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
        weldx.core.ExternalFile :
            An instance of the 'weldx.core.ExternalFile' type.

        """

        return ExternalFile(None, _tree=tree)
