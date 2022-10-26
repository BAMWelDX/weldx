"""Contains classes for the asdf serialization of media files."""

from weldx.asdf.types import WeldxConverter
from weldx.util.external_file import ExternalFile
from weldx.util.media_file import MediaFile


class MediaFileConverter(WeldxConverter):
    """Serialization class for `weldx.util.MediaFile`."""

    name = "core/media_file"
    version = "0.1.0"
    types = [MediaFile]

    def to_yaml_tree(self, obj: MediaFile, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = dict(
            file=obj.file(),
            reference_time=obj.reference_time,
            fps=obj.fps,
            n_frames=len(obj),
            resolution=obj.resolution,
        )
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        file: ExternalFile = node["file"]
        fps = node["fps"]
        reference_time = node["reference_time"]
        result = MediaFile(file.path, fps=fps, reference_time=reference_time)
        return result
