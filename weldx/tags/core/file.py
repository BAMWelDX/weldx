"""Contains classes for the asdf serialization of an external file."""

from copy import deepcopy

import numpy as np

from weldx.asdf.types import WeldxConverter
from weldx.exceptions import WeldxException
from weldx.util.external_file import ExternalFile

# Python class -------------------------------------------------------------------------


# ASDF Serialization -------------------------------------------------------------------


class ExternalFileConverter(WeldxConverter):
    """Serialization class for `weldx.core.ExternalFile`."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/file-0.1.*"]
    types = [ExternalFile]

    def to_yaml_tree(self, obj: ExternalFile, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = deepcopy(obj.__dict__)

        path = tree.pop("path", None)
        buffer = tree.pop("buffer", None)
        save_content = tree.pop("asdf_save_content")
        algorithm = tree.pop("hashing_algorithm")
        hash_value = tree.pop("hash")

        if save_content:
            if buffer is None:
                buffer = obj.get_file_content()
            tree["content"] = np.frombuffer(buffer, dtype=np.uint8)

        if buffer is None:
            if hash_value is None and path is not None:
                hash_value = obj.calculate_hash(path, obj.hashing_algorithm)
        else:
            hash_value = obj.calculate_hash(buffer, obj.hashing_algorithm)

        if hash_value is not None:
            tree["content_hash"] = {"algorithm": algorithm, "value": hash_value}
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        buffer = node.pop("content", None)
        if buffer is not None:
            buffer = buffer.tobytes()
            node["buffer"] = buffer

        hash_data = node.pop("content_hash", None)
        if hash_data is not None:
            node["hashing_algorithm"] = hash_data["algorithm"]
            node["hash"] = hash_data["value"]

        if buffer is not None:
            hash_buffer = ExternalFile.calculate_hash(buffer, node["hashing_algorithm"])
            if hash_buffer != node["hash"]:  # pragma: no cover
                raise WeldxException(
                    "The stored hash does not match the stored contents' hash."
                )
        return ExternalFile(**node)
