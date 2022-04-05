"""Media file."""
import contextlib
import typing
from pathlib import Path
from typing import Optional, Union, get_args

import numpy as np
import pandas as pd
import pint
import xarray as xr
from numpy.typing import ArrayLike

from weldx import Q_
from weldx.tags.core.file import ExternalFile
from weldx.types import types_path_like

types_media_input = Union[types_path_like, ArrayLike]


@contextlib.contextmanager
def _closeable_video_capture(file_name):
    import cv2

    cap = cv2.VideoCapture(str(file_name))

    yield cap
    cap.release()


class MediaFile:
    """Should support both external files (dirs???) and in-memory data."""

    def __init__(self, path_or_array: types_media_input, reference_time=None, fps=None):
        if isinstance(path_or_array, get_args(types_path_like)):
            from dask_image.imread import imread

            self._handle = imread(path_or_array)
            self._metadata = self._get_video_properties(path_or_array)
            self._wrap_data_array(path_or_array)

            if reference_time is None:
                self._reference_time = pd.Timestamp(
                    Path(path_or_array).stat().st_mtime_ns
                )
            self._from_file = True
        elif isinstance(path_or_array, list):
            self._handle = path_or_array
            if fps is None:
                raise ValueError(
                    "fps is needed to determine lengths," " but was not given."
                )
            from PIL.Image import fromarray

            image = fromarray(self._handle[0])
            self._metadata = dict(
                fps=fps,
                resolution=(image.width, image.height),
                nframes=len(path_or_array),
            )
            if reference_time is None:
                self._reference_time = pd.Timestamp(-1)
            self._from_file = False
            self._wrap_data_array("from_buffer")
        else:
            raise ValueError(f"unsupported input: {path_or_array}")

        self._path_or_array = path_or_array

    def _wrap_data_array(self, path_or_array):
        t_s = np.linspace(0, self.duration.m, len(self._handle))

        da = xr.DataArray(self._handle, name=path_or_array).rename(
            dict(dim_0="frames", dim_1="height", dim_2="width", dim_3="color")
        )
        self._array = da.assign_coords(frames=t_s)
        self._array.frames.attrs["units"] = "s"

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
    def reference_time(self) -> pd.Timestamp:
        """Time of recording this media."""
        # TODO: EXIF tag, plain m-time?
        return self._reference_time

    @property
    def attrs(self):
        """Video attributes."""
        from attrdict import AttrDict

        return AttrDict(self._metadata)

    @property
    def resolution(self) -> typing.Tuple[int, int]:
        """Resolution in pixels (widths, height)."""
        return self._metadata["resolution"]

    @property
    def fps(self) -> pint.Quantity:
        """Frames per second."""
        return Q_(self._metadata["fps"], "1/s")

    @property
    def duration(self) -> Optional[pint.Quantity]:
        """In case of time-dynamic data, return its duration."""
        return len(self._handle) / self.fps

    def file(self) -> ExternalFile:
        """File reference to underlying file/directory."""
        # TODO: this will rehash every time we serialize
        # even if the underlying path is unchanged.
        if not self._from_file:
            one_buff = bytes(np.array(self._path_or_array))
            return ExternalFile(
                buffer=one_buff, filename="<in-memory-source>", asdf_save_content=True
            )
        else:
            return ExternalFile(self._path_or_array)

    @property
    def data(self) -> xr.DataArray:
        """Get underlying DataArray."""
        return self._array

    def __getitem__(self, item):
        """Delegate slicing etc. to array."""
        return self._array.__getitem__(item)

    def __len__(self):
        """Length."""
        return len(self._array)
