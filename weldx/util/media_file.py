"""Media file."""
from typing import Union

import numpy as np
import pandas as pd

from weldx.types import types_path_like

types_media_input = Union[types_path_like, np.ndarray]


class MediaFile:
    """Should support both external files (dirs???) and in-memory data."""

    def __init__(self, path_or_array: types_media_input):
        if isinstance(path_or_array, types_path_like.__args__):
            import pims

            self._handle = pims.open(path_or_array)
            self._fps = NotImplemented
            self._recored_at = NotImplemented
        elif isinstance(path_or_array, np.ndarray):
            self._handle = path_or_array
            self._fps = None
            self._recored_at = NotImplemented
        else:
            raise ValueError(f"unsupported input: {path_or_array}")

    @property
    def recorded_at(self) -> pd.Timestamp:
        """Time of recording this media."""
        # TODO: EXIF tag, plain m-time?
        return self._recorded_at

    @property
    def fps(self):
        """Frames per second."""
        return self._fps

    def by_time(self, timestamps):
        """Select frames, images by given timestamps (slice?)."""
        # TODO: translate time stamps to frame indices
        raise NotImplementedError

    def __getitem__(self, item):
        """Delegate slicing etc. to handle."""
        # TODO: this interface is super raw, if we want indexing by timestamps.
        # this would fail.
        return self._handle.__getitem__(item)
