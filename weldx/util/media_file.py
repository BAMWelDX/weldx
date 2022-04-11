"""Media file."""
import typing
from pathlib import Path
from typing import Optional, Sequence, Union, get_args

import numpy as np
import pandas as pd
import pint
import xarray as xr

from weldx import Q_
from weldx.tags.core.file import ExternalFile
from weldx.types import types_path_like

types_media_input = Union[
    types_path_like,
    Sequence[Sequence[int]],
]


# _pts_to_frame, _get_frame_rate, _get_frame_count taken from
# https://github.com/PyAV-Org/PyAV/blob/v9.1.1/scratchpad/frame_seek_example.py


def _pts_to_frame(pts, time_base, frame_rate, start_time):
    return int(pts * time_base * frame_rate) - int(start_time * time_base * frame_rate)


def _get_frame_rate(stream):
    if stream.average_rate.denominator and stream.average_rate.numerator:
        return float(stream.average_rate)

    if stream.time_base.denominator and stream.time_base.numerator:
        return 1.0 / float(stream.time_base)
    raise ValueError("Unable to determine FPS")


def _get_frame_count(f, stream):
    AV_TIME_BASE = 1000000

    if stream.frames:
        return stream.frames
    if stream.duration:
        return _pts_to_frame(
            stream.duration, float(stream.time_base), _get_frame_rate(stream), 0
        )
    if f.duration:
        return _pts_to_frame(
            f.duration, 1 / float(AV_TIME_BASE), _get_frame_rate(stream), 0
        )


class MediaFile:
    """Encapsulates a media file, like video or image stacks making them accessible.

    The underlying images are encapsulated to be loaded lazily (via Dask) and can be
    accessed by a time coordinate in case of videos.


    Examples
    --------
    tODO: where to get example data?

    Parameters
    ----------
    path_or_array :
        Path pointing towards a video, or tiff stack. If an array is given, it will be
        treated as a video.
    reference_time :
        A reference time, when this media was recorded. Useful, when passing arrays.
    fps :
        Frames per second in case of a video. Has to be passed in case `path_or_array`
        was given as list of frames.
    """

    def __init__(
        self,
        path_or_array: types_media_input,
        reference_time: pd.Timestamp = None,
        fps: float = None,
    ):
        if isinstance(path_or_array, get_args(types_path_like)):
            from dask_image.imread import imread

            path = Path(path_or_array)  # type: ignore[arg-type]

            self._handle = imread(path_or_array)
            self._metadata = self._get_video_properties(str(path))
            self._wrap_data_array(path_or_array)

            if reference_time is None:
                self._reference_time = pd.Timestamp(path.stat().st_mtime_ns)
            self._from_file = True
        elif isinstance(path_or_array, list):
            self._handle = path_or_array
            if fps is None:
                raise ValueError(
                    "fps is needed to determine duration, but was not given."
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
    def _get_video_properties(fn: str) -> dict:
        import av

        v = av.open(fn)
        frame = next(v.decode())
        resolution = frame.width, frame.height

        stream = next(s for s in v.streams if s.type == "video")

        metadata = dict(
            fps=_get_frame_rate(stream),
            nframes=_get_frame_count(
                frame,
                stream,
            ),
            resolution=resolution,
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
            return ExternalFile(self._path_or_array)  # type: ignore[arg-type]

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
