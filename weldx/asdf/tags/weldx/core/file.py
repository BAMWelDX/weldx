"""Contains the asdf serialization class for `weldx.core.ExternalFileBuffer`."""


from copy import deepcopy

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
        # hash_algorithm = "SHA-256"
        # if node.asdf_save_content:
        # buffer = node.get_file_content()
        # buffer_hash = node.calculate_hash_of_buffer(buffer, hash_algorithm)
        # buffer_np = np.frombuffer(buffer, dtype=np.uint8)
        # return {
        # "filename": node.filename,
        # "content": buffer_np,
        # "content_hash": {"algorithm": hash_algorithm, "value": buffer_hash},
        #    }
        tree = deepcopy(node.__dict__)

        path = tree.pop("path", None)
        buffer = tree.pop("buffer", None)
        save_content = tree.pop("asdf_save_content")
        algorithm = tree.pop("hashing_algorithm")

        if save_content:
            if buffer is None:
                buffer = node.get_file_content()
            tree["content"] = np.frombuffer(buffer, dtype=np.uint8)

        if buffer is None:
            hash_value = node.calculate_hash_of_file(path, node.hashing_algorithm)
        else:
            hash_value = node.calculate_hash_of_buffer(buffer, node.hashing_algorithm)
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

        hash_data = tree.pop("content_hash")
        tree["hashing_algorithm"] = hash_data["algorithm"]
        hash_stored = hash_data["value"]

        if buffer is not None:
            hash_buffer = ExternalFile.calculate_hash_of_buffer(
                buffer, tree["hashing_algorithm"]
            )
            if hash_buffer != hash_stored:  # pragma: no cover
                raise Exception(
                    "The stored hash does not match the stored contents' hash."
                )
        return ExternalFile(**tree)
