"""Media file."""

from __future__ import annotations

from pathlib import Path
from typing import Union, get_args

import numpy as np
import pandas as pd
import pint
import xarray as xr

from weldx import Q_
from weldx.types import types_path_like
from weldx.util.external_file import ExternalFile

__all__ = ["types_media_input", "MediaFile", "UnknownFormatError"]


types_sequence_like = Union[
    xr.DataArray,
    np.ndarray,
    list,
    tuple,
]

types_media_input = Union[
    types_path_like,
    types_sequence_like,
]

# _pts_to_frame, _get_frame_rate, _get_frame_count taken from
# https://github.com/PyAV-Org/PyAV/blob/v9.1.1/scratchpad/frame_seek_example.py
_AV_TIME_BASE = 1000000


def _pts_to_frame(pts, time_base, frame_rate, start_time):
    return int(pts * time_base * frame_rate) - int(start_time * time_base * frame_rate)


def _get_frame_rate(stream):
    if stream.average_rate.denominator and stream.average_rate.numerator:
        return float(stream.average_rate)

    if stream.time_base.denominator and stream.time_base.numerator:
        return 1.0 / float(stream.time_base)
    raise ValueError("Unable to determine FPS")


def _get_frame_count(f, stream):
    if stream.frames:
        return stream.frames
    if stream.duration:
        return _pts_to_frame(
            stream.duration, float(stream.time_base), _get_frame_rate(stream), 0
        )
    if f.duration:
        return _pts_to_frame(
            f.duration, 1 / float(_AV_TIME_BASE), _get_frame_rate(stream), 0
        )
    return float("nan")


class UnknownFormatError(Exception):
    """File format could not be determined."""


class MediaFile:
    """Encapsulates a media file, like video or image stacks making them accessible.

    The underlying images are encapsulated to be loaded lazily (via Dask) and can be
    accessed by a time coordinate in case of videos.

    Parameters
    ----------
    path_or_array :
        Path pointing towards a video, or tiff stack. If an array is given, it will be
        treated as a video.
    reference_time :
        A reference time, when this media was recorded. Useful, when passing arrays.
    fps :
        Frames per second in case of a video. Has to be passed in case ``path_or_array``
        was given as list of frames.
    """

    def __init__(
        self,
        path_or_array: types_media_input,
        reference_time: pd.Timestamp | None = None,
        fps: float | None = None,
    ):
        if isinstance(path_or_array, get_args(types_path_like)):
            self._init_from_path(path_or_array, reference_time)  # type: ignore
        elif isinstance(path_or_array, get_args(types_sequence_like)):
            self._init_from_sequence(fps, path_or_array, reference_time)  # type: ignore
        else:
            raise ValueError(f"unsupported input: {path_or_array}")

        self._path_or_array = path_or_array

    def _init_from_path(self, path_: types_path_like, reference_time):
        from dask_image.imread import imread
        from pims import UnknownFormatError as _UnknownFormatError

        path = Path(path_)  # type: ignore[arg-type]
        try:
            self._handle = imread(path_)
        except _UnknownFormatError as e:
            # wrap in our exception type to hide impl detail.
            raise UnknownFormatError(e) from e
        self._metadata = self._get_video_metadata(str(path))
        self._wrap_data_array(path_)
        if reference_time is None:
            self._reference_time = pd.Timestamp(path.stat().st_mtime_ns)
        elif isinstance(reference_time, pd.Timestamp):
            self._reference_time = reference_time
        else:
            raise ValueError(
                f"unsupported type for reference_time {type(reference_time)}"
            )
        self._from_file = True

    def _init_from_sequence(
        self, fps, path_or_array: types_sequence_like, reference_time
    ):
        self._handle = path_or_array
        if fps is None:
            raise ValueError("fps is needed to determine duration, but was not given.")
        from PIL.Image import fromarray

        first_frame = self._handle[0]
        if not hasattr(first_frame, "__array_interface__"):
            first_frame = first_frame.data  # type: ignore
        image = fromarray(first_frame)
        self._metadata = dict(
            fps=fps,
            resolution=(image.width, image.height),
            nframes=len(path_or_array),
        )
        if reference_time is not None and not isinstance(reference_time, pd.Timestamp):
            raise ValueError(
                f"unsupported type for reference_time {type(reference_time)}"
            )
        self._reference_time = reference_time
        self._from_file = False
        self._wrap_data_array(array_name="from_buffer")

    def _wrap_data_array(self, array_name):
        if isinstance(self._handle, xr.DataArray):
            self._array = self._handle  # TODO: this is kinda ugly!
        else:
            t_s = np.linspace(0, self.duration.m, len(self._handle))

            da = xr.DataArray(self._handle, name=array_name).rename(
                dict(dim_0="frames", dim_1="height", dim_2="width", dim_3="color")
            )
            self._array = da.assign_coords(frames=t_s)
            self._array.frames.attrs["units"] = "s"

    @property
    def from_file(self) -> bool:
        """Initialized from file or not?"""
        return self._from_file

    @staticmethod
    def _get_video_metadata(fn: str) -> dict:
        import av

        with av.open(fn) as v:
            frame = next(v.decode(), None)
            if not frame:
                raise RuntimeError(
                    "could not determine video metadata, "
                    "as no single frame could be read."
                )
            resolution = frame.width, frame.height

            stream = next((s for s in v.streams if s.type == "video"), None)

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
    def reference_time(self) -> pd.Timestamp | None:
        """Time of recording of this media (if known)."""
        return self._reference_time

    @property
    def attrs(self):
        """Video attributes."""
        from attrdict import AttrDict

        return AttrDict(self._metadata)

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution in pixels (widths, height)."""
        return self._metadata["resolution"]

    @property
    def fps(self) -> pint.Quantity:
        """Frames per second."""
        return Q_(self._metadata["fps"], "1/s")

    @property
    def duration(self) -> pint.Quantity | None:
        """In case of time-dynamic data, return its duration."""
        return len(self._handle) / self.fps

    def file(self) -> ExternalFile:
        """File reference to underlying file/directory."""
        # note: this will rehash every time we serialize
        # even if the underlying path is unchanged.
        if not self._from_file:
            buffer = bytes(np.array(self._path_or_array))
            return ExternalFile(
                buffer=buffer, filename="<in-memory-source>", asdf_save_content=True
            )
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
