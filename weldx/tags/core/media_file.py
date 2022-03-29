"""Contains classes for the asdf serialization of media files."""

from weldx.asdf.types import WeldxConverter
from weldx.util.media_file import MediaFile


class MediaFileConverter(WeldxConverter):
    """Serialization class for `weldx.util.MediaFile`."""

    name = "core/media_file"
    version = "0.1.0"
    types = [MediaFile]

    def to_yaml_tree(self, obj: MediaFile, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = obj.__dict__
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        raise NotImplementedError
