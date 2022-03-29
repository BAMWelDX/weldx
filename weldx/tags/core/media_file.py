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
        tree = dict(file=obj.file(), recorded_at=obj.recorded_at)
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        from weldx.tags.core.file import ExternalFile

        file: ExternalFile = node["file"]
        result = MediaFile(file.path)
        assert result.recorded_at == node["recorded_at"]
        return result
