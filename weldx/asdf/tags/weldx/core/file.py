"""Contains the asdf serialization class for `weldx.core.ExternalFileBuffer`."""

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
        return {"filename": node.filename, "content": node.buffer}

    @classmethod
    def from_tree(cls, tree, ctx):

        return ExternalFileBuffer(tree["filename"], buffer=tree["content"])
