"""Media file."""
import contextlib
from pathlib import Path
from typing import Optional, Union, get_args

import numpy as np
import pandas as pd
import pint
import xarray as xr

from weldx import Q_
from weldx.tags.core.file import ExternalFile
from weldx.types import types_path_like

types_media_input = Union[types_path_like, np.ndarray]


@contextlib.contextmanager
def _closeable_video_capture(file_name):
    import cv2

    cap = cv2.VideoCapture(str(file_name))

    yield cap
    cap.release()


class MediaFile:
    """Should support both external files (dirs???) and in-memory data."""

    def __init__(self, path_or_array: types_media_input):
        if isinstance(path_or_array, get_args(types_path_like)):
            from dask_image.imread import imread

            self._handle = imread(path_or_array)
            self._metadata = self._get_video_properties(path_or_array)

            length_in_seconds = len(self._handle) / self.fps
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

    @staticmethod
    def _get_video_properties(fn) -> dict:
        import cv2

        with _closeable_video_capture(fn) as cap:
            metadata = dict(
                fps=cap.get(cv2.CAP_PROP_FPS),
                resolution=(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                ),
                nframes=cap.get(cv2.CAP_PROP_POS_FRAMES),
            )
        return metadata

    @property
    def recorded_at(self) -> pd.Timestamp:
        """Time of recording this media."""
        # TODO: EXIF tag, plain m-time?
        return self._recorded_at

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._metadata["fps"]

    @property
    def duration(self) -> Optional[pint.Quantity]:
        """In case of time-dynamic data, return its duration."""
        return self._length_in_seconds

    def file(self) -> ExternalFile:
        """File reference to underlying file/directory."""
        # TODO: this will rehash every time we serialize
        # even if the underlying path is unchanged.
        if isinstance(self._path_or_array, np.ndarray):
            return ExternalFile(buffer=bytes(self._path_or_array.astype(np.int8)))
        else:
            return ExternalFile(self._path_or_array)

    @property
    def data(self) -> xr.DataArray:
        """Get underlying DataArray."""
        return self._array

    def __getitem__(self, item):
        """Delegate slicing etc. to array."""
        return self._array.__getitem__(item)
