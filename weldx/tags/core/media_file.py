"""Contains classes for the asdf serialization of media files."""

import pathlib

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
            reference_time=obj.reference_time,
            fps=obj.fps,
            n_frames=len(obj),
            resolution=obj.resolution,
        )
        if obj.from_file:
            tree["file"] = obj.file()
        else:
            tree["data"] = obj.data
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        if "data" in node:
            data = node["data"]
        elif "file" in node:
            file: ExternalFile = node["file"]
            data = pathlib.Path(file.directory) / file.filename
        else:
            raise RuntimeError("malformed media file. Lacking keys 'data' or 'file'.")
        fps = node["fps"]
        reference_time = node["reference_time"] if "reference_time" in node else None
        result = MediaFile(data, fps=fps, reference_time=reference_time)
        return result
