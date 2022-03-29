"""Media file."""
import contextlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from weldx import Q_
from weldx.tags.core.file import ExternalFile
from weldx.types import types_path_like

types_media_input = Union[types_path_like, np.ndarray]


@contextlib.contextmanager
def _closeable_video_capture(file_name):
    import cv2

    cap = cv2.VideoCapture(file_name)

    yield cap
    cap.release()


class MediaFile:
    """Should support both external files (dirs???) and in-memory data."""

    def __init__(self, path_or_array: types_media_input):
        if isinstance(path_or_array, types_path_like.__args__):
            from dask_image.imread import imread

            self._handle = imread(path_or_array)
            self._fps = self._get_fps(path_or_array)

            length_in_seconds = len(self._handle) / self._fps
            t_s = np.linspace(0, length_in_seconds, len(self._handle))
            self._length_in_seconds = length_in_seconds * Q_("s")

            da = xr.DataArray(self._handle, name=path_or_array).rename(
                dict(dim_0="frames", dim_1="height", dim_2="width", dim_3="rgb")
            )
            self._array = da.assign_coords(frames=t_s)
            self._array.frames.attrs["units"] = "s"
            self._recorded_at = pd.Timestamp(Path(path_or_array).stat().st_mtime_ns)
        elif isinstance(path_or_array, np.ndarray):
            self._handle = path_or_array
            self._fps = None
            self._recorded_at = NotImplemented
        else:
            raise ValueError(f"unsupported input: {path_or_array}")

        self._path_or_array = path_or_array

    def _get_fps(self, fn) -> float:
        import cv2

        with _closeable_video_capture(fn) as cap:
            # TODO: we have a plethora of properties from opencv.
            # expose them in a sane way
            # and further serialize them in a dict like structure.
            fps = cap.get(cv2.CAP_PROP_FPS)

        return fps

    @property
    def recorded_at(self) -> pd.Timestamp:
        """Time of recording this media."""
        # TODO: EXIF tag, plain m-time?
        return self._recorded_at

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def duration(self) -> Optional[Q_]:
        """In case of time-dynamic data, return its duration."""
        return self._length_in_seconds

    def file(self) -> ExternalFile:
        """File reference to underlying file/directory."""
        if isinstance(self._path_or_array, np.ndarray):
            return ExternalFile(buffer=self._path_or_array.astype(bytes))
        else:
            return ExternalFile(self._path_or_array)

    @property
    def data(self) -> xr.DataArray:
        """Get underlying DataArray."""
        return self._array

    def __getitem__(self, item):
        """Delegate slicing etc. to array."""
        return self._array.__getitem__(item)
