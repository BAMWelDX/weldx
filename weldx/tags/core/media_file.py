"""Contains classes for the asdf serialization of media files."""

from dataclasses import dataclass

import pandas as pd

from weldx import Time
from weldx.asdf.types import WeldxConverter

from weldx.tags.core.file import ExternalFile


@dataclass
class MediaFile:
    """Handles the asdf serialization of media files."""

    file: ExternalFile = None
    fps: Time = None
    captured_at: pd.Timestamp = None

    def __post_init__(self):
        """Initialize the internal values."""


class ExternalFileConverter(WeldxConverter):
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
