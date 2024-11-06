"""Contains methods and classes for coordinate transformations."""

from __future__ import annotations

import typing
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation as Rot

import weldx.util as ut
from weldx.constants import _DEFAULT_LEN_UNIT, Q_
from weldx.core import TimeSeries
from weldx.exceptions import WeldxException
from weldx.time import Time, TimeDependent, types_time_like, types_timestamp_like
from weldx.transformations.types import (
    types_coordinates,
    types_homogeneous,
    types_orientation,
)
from weldx.transformations.util import normalize
from weldx.types import UnitLike

__all__ = ("LocalCoordinateSystem",)

if typing.TYPE_CHECKING:
    import matplotlib  # noqa: ICN001


class LocalCoordinateSystem(TimeDependent):
    """Defines a local cartesian coordinate system in 3d.

    Notes
    -----
    Learn how to use this class by reading the
    tutorial <tutorials/transformations_01_coordinate_systems>.

    """

    def __init__(
        self,
        orientation: types_orientation = None,
        coordinates: types_coordinates | TimeSeries = None,
        time: types_time_like = None,
        time_ref: types_timestamp_like = None,
        construction_checks: bool = True,
    ):
        """Construct a cartesian coordinate system.

        Parameters
        ----------
        orientation :
            Matrix of 3 orthogonal column vectors which represent
            the coordinate systems orientation. Keep in mind, that the columns of the
            corresponding orientation matrix is equal to the normalized orientation
            vectors. So each orthogonal transformation matrix can also be
            provided as orientation.
            Passing a scipy.spatial.transform.Rotation object is also supported.
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems. If the provided coordinates
            and orientations contain only a single value, the coordinate system is
            considered to be static and the provided value won't be stored. If this
            happens, a warning will be emitted.
        time_ref :
            Optional reference timestamp if ``time`` is a time delta.
        construction_checks :
            If 'True', the validity of the data will be verified

        Returns
        -------
        LocalCoordinateSystem
            Cartesian coordinate system

        """
        time_cls = self._build_time(coordinates, time, time_ref)
        orientation = self._build_orientation(orientation, time_cls)
        coordinates = self._build_coordinates(coordinates, time_cls)

        # warn about dropped time data
        if (
            time is not None
            and len(Time(time)) > 1
            and "time" not in orientation.dims
            and (isinstance(coordinates, TimeSeries) or "time" not in coordinates.dims)
        ):
            warnings.warn(
                "Provided time is dropped because of the given coordinates and "
                "orientation.",
                stacklevel=2,
            )

        if construction_checks:
            self._check_coordinates(coordinates)
            orientation = self._check_and_normalize_orientation(orientation)

        orientation, coordinates = self._unify_time_axis(orientation, coordinates)

        orientation.name = "orientation"
        dataset_items = [orientation]

        self._coord_ts = None
        if isinstance(coordinates, TimeSeries):
            self._coord_ts = coordinates
        else:
            coordinates.name = "coordinates"
            dataset_items.append(coordinates)

        self._dataset = xr.merge(dataset_items, join="exact")

        self._time_ref = time_cls.reference_time if isinstance(time_cls, Time) else None
        if "time" in self._dataset and self._time_ref is not None:
            self._dataset.weldx.time_ref = self._time_ref

        # warn about dropped reference time
        if time_ref is not None and time is None and self._time_ref is None:
            warnings.warn(
                "Reference time dropped. The system is not time dependent.",
                stacklevel=2,
            )

    def __repr__(self):
        """Give __repr_ output in xarray format."""
        # todo: rewrite if expressions are fully supported
        return self._dataset.__repr__().replace(
            "<xarray.Dataset", "<LocalCoordinateSystem"
        )

    def __add__(self, rhs_cs: LocalCoordinateSystem) -> LocalCoordinateSystem:
        """Add 2 coordinate systems.

        Generates a new coordinate system by treating the left-hand side
        coordinate system as being defined in the right hand-side coordinate
        system.
        The transformations from the base coordinate system to the new
        coordinate system are equivalent to the combination of the
        transformations from both added coordinate systems:

        R_n = R_r * R_l
        T_n = R_r * T_l + T_r

        R_r and T_r are rotation matrix and translation vector of the
        right-hand side coordinate system, R_l and T_l of the left-hand side
        coordinate system and R_n and T_n of the resulting coordinate system.

        If the left-hand side system has a time component, the data of the right-hand
        side system will be interpolated to the same times, before the previously shown
        operations are performed per point in time. In case, that the left-hand side
        system has no time component, but the right-hand side does, the resulting system
        has the same time components as the right-hand side system.

        Parameters
        ----------
        rhs_cs :
            Right-hand side coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Resulting coordinate system.

        """
        lhs_cs = self

        if isinstance(lhs_cs.coordinates, TimeSeries) or isinstance(
            rhs_cs.coordinates, TimeSeries
        ):
            raise WeldxException(
                "Addition of coordinate systems that use a 'TimeSeries' as coordinates "
                "is not supported. Use 'interp_time' to create discrete values."
            )

        # handle time
        # TODO: time var is unused! why? The var should be used in line 201?
        # e.g. ... rhs_cs.interp_time(lhs_cs.time, time_ref)
        time = lhs_cs.time
        if time is None:
            time = rhs_cs.time

        # handle reference times
        if (
            lhs_cs.reference_time != rhs_cs.reference_time
            and lhs_cs.has_reference_time
            and rhs_cs.has_reference_time
        ):
            if lhs_cs.reference_time < rhs_cs.reference_time:
                time_ref = lhs_cs.reference_time
                rhs_cs = deepcopy(rhs_cs)
                rhs_cs.reset_reference_time(time_ref)
            else:
                time_ref = rhs_cs.reference_time
                lhs_cs = deepcopy(lhs_cs)
                lhs_cs.reset_reference_time(time_ref)
        elif not lhs_cs.has_reference_time:
            time_ref = rhs_cs.reference_time
        else:
            time_ref = lhs_cs.reference_time

        # interpolate rhs time to match lhs
        rhs_cs = rhs_cs.interp_time(lhs_cs.time, time_ref)

        # calculate resulting orientation and coordinates
        orientation = ut.xr_matmul(
            rhs_cs.orientation, lhs_cs.orientation, dims_a=["c", "v"]
        )
        coordinates = (
            ut.xr_matmul(rhs_cs.orientation, lhs_cs.coordinates, ["c", "v"], ["c"])
            + rhs_cs.coordinates
        )
        return LocalCoordinateSystem(orientation, coordinates, time_ref=time_ref)

    def __sub__(self, rhs_cs: LocalCoordinateSystem) -> LocalCoordinateSystem:
        """Subtract 2 coordinate systems.

        Generates a new coordinate system from two local coordinate systems
        with the same reference coordinate system. The resulting system is
        equivalent to the left-hand side system but with the right-hand side
        as reference coordinate system.
        This is achieved by the following transformations:

        R_n = R_r^(-1) * R_l
        T_n = R_r^(-1) * (T_l - T_r)

        R_r and T_r are rotation matrix and translation vector of the
        right-hand side coordinate system, R_l and T_l of the left-hand side
        coordinate system and R_n and T_n of the resulting coordinate system.

        If the left-hand side system has a time component, the data of the right-hand
        side system will be interpolated to the same times, before the previously shown
        operations are performed per point in time. In case, that the left-hand side
        system has no time component, but the right-hand side does, the resulting system
        has the same time components as the right-hand side system.

        Parameters
        ----------
        rhs_cs :
            Right-hand side coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Resulting coordinate system.

        """
        rhs_cs_inv = rhs_cs.invert()
        return self + rhs_cs_inv

    def __eq__(self, other: Any) -> bool:
        """Check equality of LocalCoordinateSystems."""
        if not isinstance(other, LocalCoordinateSystem):
            return NotImplemented

        def _comp_coords():
            if not isinstance(self.coordinates, type(other.coordinates)):
                return False
            if isinstance(self.coordinates, TimeSeries):
                return self.coordinates == other.coordinates
            return self.coordinates.identical(other.coordinates)

        return (
            self.orientation.identical(other.orientation)
            and _comp_coords()
            and self.reference_time == other.reference_time
        )

    __hash__ = None

    @staticmethod
    def _build_orientation(
        orientation: types_orientation,
        time: Time = None,
    ) -> xr.DataArray:
        """Create xarray orientation from different formats and time-inputs."""
        if orientation is None:
            orientation = np.eye(3)

        if not isinstance(orientation, xr.DataArray):
            if isinstance(orientation, Rot):
                orientation = orientation.as_matrix()
            elif not isinstance(orientation, np.ndarray):
                orientation = np.array(orientation)
            orientation = ut.xr_3d_matrix(orientation, time)

        # make sure we have correct "time" format
        orientation = orientation.weldx.time_ref_restore()

        return orientation

    @classmethod
    def _build_coordinates(
        cls, coordinates, time: Time = None
    ) -> xr.DataArray | TimeSeries:
        """Create xarray coordinates from different formats and time-inputs."""
        if isinstance(coordinates, TimeSeries):
            if coordinates.is_expression:
                if not coordinates.units.is_compatible_with(_DEFAULT_LEN_UNIT):
                    raise pint.DimensionalityError(
                        coordinates.units,
                        _DEFAULT_LEN_UNIT,
                        extra_msg="\nThe units resulting from the expression must "
                        "represent a length.",
                    )
                return coordinates
            coordinates = cls._coords_from_discrete_time_series(coordinates)

        if coordinates is None:
            coordinates = Q_(np.zeros(3), _DEFAULT_LEN_UNIT)

        if not isinstance(coordinates, xr.DataArray):
            if not isinstance(coordinates, (np.ndarray, pint.Quantity)):
                coordinates = np.array(coordinates)

            coordinates = ut.xr_3d_vector(coordinates, time)

        c_data = coordinates.data
        if not isinstance(c_data, pint.Quantity) or not c_data.is_compatible_with(
            _DEFAULT_LEN_UNIT
        ):
            unit = c_data.u if isinstance(c_data, pint.Quantity) else None
            raise pint.DimensionalityError(
                unit,
                _DEFAULT_LEN_UNIT,
                extra_msg="\nThe coordinates require units representing a length.",
            )

        # make sure we have correct "time" format
        coordinates = coordinates.weldx.time_ref_restore()

        return coordinates

    @staticmethod
    def _build_time(
        coordinates: types_coordinates | TimeSeries,
        time: types_time_like,
        time_ref: types_timestamp_like,
    ) -> Time | None:
        if time is None:
            if isinstance(coordinates, TimeSeries) and coordinates.is_discrete:
                time = coordinates.time
            # this branch is relevant if coordinates and orientations are xarray types
            elif time_ref is not None:
                time = time_ref

        return Time(time, time_ref) if time is not None else None

    @staticmethod
    def _check_and_normalize_orientation(orientation: xr.DataArray) -> xr.DataArray:
        """Check if the orientation has the correct format and normalize it."""
        ut.xr_check_coords(
            orientation,
            dict(
                c={"values": ["x", "y", "z"]},
                v={"values": [0, 1, 2]},
                time={"dtype": "timedelta64", "optional": True},
            ),
        )

        orientation = xr.apply_ufunc(
            normalize,
            orientation,
            input_core_dims=[["c"]],
            output_core_dims=[["c"]],
            keep_attrs=True,
        )

        # vectorize test if orthogonal
        if not ut.xr_is_orthogonal_matrix(orientation, dims=["c", "v"]):
            raise ValueError("Orientation vectors must be orthogonal")

        return orientation

    @staticmethod
    def _check_coordinates(coordinates: xr.DataArray | TimeSeries):
        """Check if the coordinates have the correct format."""
        if isinstance(coordinates, xr.DataArray):
            ut.xr_check_coords(
                coordinates,
                dict(
                    c={"values": ["x", "y", "z"]},
                    time={"dtype": "timedelta64", "optional": True},
                ),
            )
        else:
            # todo: check time series shape
            pass

    @staticmethod
    def _coords_from_discrete_time_series(time_series):
        """Return compatible coordinates from a discrete `TimeSeries`."""
        if time_series.shape[1] != 3:
            raise ValueError(
                "The shape of the TimeSeries must be (n, 3). It actually is: "
                f"{time_series.shape}"
            )
        coordinates = time_series.data_array

        c_dict = dict(c=["x", "y", "z"])
        if coordinates.data.shape[0] == 1:
            return xr.DataArray(coordinates.data.reshape(3), dims=["c"], coords=c_dict)
        return coordinates.rename({coordinates.dims[1]: "c"}).assign_coords(c_dict)

    @staticmethod
    def _unify_time_axis(
        orientation: xr.DataArray, coordinates: xr.DataArray | TimeSeries
    ) -> tuple:
        """Unify time axis of orientation and coordinates if both are DataArrays."""
        if (
            not isinstance(coordinates, TimeSeries)
            and ("time" in orientation.dims)
            and ("time" in coordinates.dims)
            and (not np.all(orientation.time.data == coordinates.time.data))
        ):
            time_union = Time.union([orientation.time, coordinates.time])
            orientation = ut.xr_interp_orientation_in_time(orientation, time_union)
            coordinates = ut.xr_interp_coordinates_in_time(coordinates, time_union)

        return orientation, coordinates

    @classmethod
    def from_euler(
        cls,
        sequence,
        angles,
        degrees=False,
        coordinates=None,
        time: types_time_like = None,
        time_ref: types_timestamp_like = None,
    ) -> LocalCoordinateSystem:
        """Construct a local coordinate system from an euler sequence.

        This function uses scipy.spatial.transform.Rotation.from_euler method to define
        the coordinate systems orientation. Take a look at its documentation, if some
        information is missing here. The related parameter docs are a copy of the scipy
        documentation.

        Parameters
        ----------
        sequence :
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations,
            or {‘x’, ‘y’, ‘z’} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call.
        angles :
            Euler angles specified in radians (degrees is False) or degrees
            (degrees is True). For a single character seq, angles can be:
            - a single value
            - array_like with shape (N,), where each angle[i] corresponds to a single
            rotation
            - array_like with shape (N, 1), where each angle[i, 0] corresponds to a
            single rotation
            For 2- and 3-character wide seq, angles can be:
            - array_like with shape (W,) where W is the width of seq, which corresponds
            to a single rotation with W axes
            - array_like with shape (N, W) where each angle[i] corresponds to a sequence
            of Euler angles describing a single rotation
        degrees :
            If True, then the given angles are assumed to be in degrees.
            Default is False.
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Optional reference timestamp if ``time`` is a time delta.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        orientation = Rot.from_euler(sequence, angles, degrees)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_axis_vectors(
        cls,
        x: types_coordinates = None,
        y: types_coordinates = None,
        z: types_coordinates = None,
        coordinates: types_coordinates = None,
        time: types_time_like = None,
        time_ref: types_timestamp_like = None,
    ) -> LocalCoordinateSystem:
        """Create a LCS from two or more coordinate axes.

        If only two axes are provided, the third one is calculated under the assumption
        of a positively oriented coordinate system (right hand system).

        Parameters
        ----------
        x :
            A vector representing the coordinate systems x-axis
        y :
            A vector representing the coordinate systems y-axis
        z :
            A vector representing the coordinate systems z-axis
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Optional reference timestamp if ``time`` is a time delta.

        Returns
        -------
        LocalCoordinateSystem
            The new coordinate system

        Examples
        --------
        Create a coordinate system from 3 orthogonal vectors:

        >>> from weldx import LocalCoordinateSystem
        >>>
        >>> x = [2, 2, 0]
        >>> y = [-8, 8, 0]
        >>> z = [0, 0, 3]
        >>>
        >>> lcs = LocalCoordinateSystem.from_axis_vectors(x, y, z)

        Create a coordinate system from 2 orthogonal vectors and let the third one be
        determined automatically:

        >>> lcs = LocalCoordinateSystem.from_axis_vectors(x=x, z=z)

        """
        mat = [x, y, z]
        num_none = sum(v is None for v in mat)

        if num_none == 1:
            idx = next(i for i, v in enumerate(mat) if v is None)  # skipcq: PTC-W0063
            # type: ignore[arg-type]
            mat[idx] = np.cross(mat[(idx - 2) % 3], mat[(idx - 1) % 3])  # type: ignore
        elif num_none > 1:
            raise ValueError("You need to specify two or more vectors.")

        mat = np.array(mat)
        t_axes = (1, 0) if mat.ndim == 2 else (1, 2, 0)
        return cls(mat.transpose(t_axes), coordinates, time, time_ref)

    @classmethod
    def from_homogeneous_transformation(
        cls,
        transformation_matrix: types_homogeneous,
        translation_unit: UnitLike,
        time: types_time_like = None,
        time_ref: types_timestamp_like = None,
    ) -> LocalCoordinateSystem:
        """Construct a local coordinate system from a homogeneous transformation matrix.

        Parameters
        ----------
        transformation_matrix :
            Describes the homogeneous transformation matrix that includes the rotation
            and the translation (coordinates).
        translation_unit :
            Unit describing the value of the translation. Necessary, because the
            homogeneous transformation matrix is unitless.
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Optional reference timestamp if ``time`` is a time delta.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        if isinstance(transformation_matrix, xr.DataArray):
            transformation_matrix = np.array(transformation_matrix.data)
        if transformation_matrix.ndim == 3:
            orientation = transformation_matrix[:, :3, :3]
            coordinates = Q_(transformation_matrix[:, :3, 3], translation_unit)
        else:
            orientation = transformation_matrix[:3, :3]
            coordinates = Q_(transformation_matrix[:3, 3], translation_unit)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @property
    def orientation(self) -> xr.DataArray:
        """Get the coordinate systems orientation matrix.

        Returns
        -------
        xarray.DataArray
            Orientation matrix

        """
        return self.dataset.orientation

    @property
    def coordinates(self) -> xr.DataArray | TimeSeries:
        """Get the coordinate systems coordinates.

        Returns
        -------
        xarray.DataArray
            Coordinates of the coordinate system

        """
        if self._coord_ts is not None:
            return self._coord_ts
        return self.dataset.coordinates

    @property
    def is_time_dependent(self) -> bool:
        """Return `True` if the coordinate system is time dependent.

        Returns
        -------
        bool :
            `True` if the coordinate system is time dependent, `False` otherwise.

        """
        return self._coord_ts is not None or ("time" in self._dataset.dims)

    @property
    def has_timeseries(self) -> bool:
        """Return `True` if the system has a `~weldx.core.TimeSeries` component."""
        return isinstance(self.coordinates, TimeSeries) or isinstance(
            self.orientation, TimeSeries
        )

    @property
    def has_reference_time(self) -> bool:
        """Return `True` if the coordinate system has a reference time.

        Returns
        -------
        bool :
            `True` if the coordinate system has a reference time, `False` otherwise.

        """
        return self.reference_time is not None

    @property
    def reference_time(self) -> pd.Timestamp | None:
        """Get the coordinate systems reference time.

        Returns
        -------
        pandas.Timestamp:
            The coordinate systems reference time

        """
        if isinstance(self.coordinates, TimeSeries):
            return self._time_ref
        return self._dataset.weldx.time_ref

    @property
    def time(self) -> Time | None:
        """Get the time union of the local coordinate system (None if system is static).

        Returns
        -------
        xarray.DataArray
            Time-like data array representing the time union of the LCS

        """
        if "time" in self._dataset.coords:
            return Time(self._dataset.time, self.reference_time)
        return None

    @property
    def dataset(self) -> xr.Dataset:
        """Get the underlying `xarray.Dataset` with ordered dimensions.

        Returns
        -------
        xarray.Dataset :
            xarray Dataset with coordinates and orientation as DataVariables.

        """
        return self._dataset.transpose(..., "c", "v")

    @property
    def is_unity_translation(self) -> bool:
        """Return true if the LCS has a zero translation/coordinates value."""
        if isinstance(self.coordinates, TimeSeries):
            return False
        coords = (
            self.coordinates.data.magnitude
            if isinstance(self.coordinates.data, pint.Quantity)
            else self.coordinates.data
        )
        return coords.shape[-1] == 3 and np.allclose(coords, np.zeros(3))

    @property
    def is_unity_rotation(self) -> bool:
        """Return true if the LCS represents a unity rotation/orientations value."""
        return self.orientation.shape[-2:] == (3, 3) and np.allclose(
            self.orientation.values, np.eye(3)
        )

    def as_euler(
        self, seq: str = "xyz", degrees: bool = False
    ) -> np.ndarray:  # pragma: no cover
        """Return Euler angle representation of the coordinate system orientation.

        Parameters
        ----------
        seq :
            Euler rotation sequence as described in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial
            .transform.Rotation.as_euler.html
        degrees :
            Returned angles are in degrees if True, else they are in radians.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of euler angles.

        """
        return self.as_rotation().as_euler(seq=seq, degrees=degrees)

    def as_rotation(self) -> Rot:  # pragma: no cover
        """Get a scipy.Rotation object from the coordinate system orientation.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Scipy rotation object representing the orientation.

        """
        return Rot.from_matrix(self.orientation.values)

    def as_homogeneous_matrix(self, translation_unit: UnitLike) -> np.ndarray:
        """Get a homogeneous transformation matrix from the coordinate system
        orientation.

        Parameters
        ----------
        translation_unit : UnitLike
            Unit the translation part of the homogeneous transformation matrix
            should represent.

        Returns
        -------
        numpy.ndarray
            Numpy array representing the homogeneous transformation matrix.

        """

        if self.is_time_dependent:
            time_dim = self.time.shape[0]
        else:
            time_dim = 1

        rotation = np.resize(self.orientation.data, (time_dim, 3, 3))
        coordinates = self.coordinates
        if not isinstance(coordinates, TimeSeries):
            translation = np.resize(
                coordinates.data.to(translation_unit).m, (time_dim, 3)
            )
            homogeneous_matrix = np.resize(np.identity(4), (time_dim, 4, 4))
            homogeneous_matrix[:, :3, :3] = rotation
            homogeneous_matrix[:, :3, 3] = translation

            return np.squeeze(homogeneous_matrix)
        else:
            raise NotImplementedError(
                "Cannot convert LCS with `TimeSeries` coordinates to homogeneous matrix"
            )

    def _interp_time_orientation(self, time: Time) -> xr.DataArray:
        """Interpolate the orientation in time."""
        if "time" not in self.orientation.dims:  # don't interpolate static
            return self.orientation
        if time.max() <= self.time.min():  # only use edge timestamp
            return self.orientation.isel(time=0).data
        if time.min() >= self.time.max():  # only use edge timestamp
            return self.orientation.isel(time=-1).data
        # full interpolation with overlapping times
        return ut.xr_interp_orientation_in_time(self.orientation, time)

    def _interp_time_coordinates(self, time: Time) -> xr.DataArray:
        """Interpolate the coordinates in time."""
        if isinstance(self.coordinates, TimeSeries):
            time_interp = Time(time, self.reference_time)
            coordinates = self._coords_from_discrete_time_series(
                self.coordinates.interp_time(time_interp)
            )
            if self.has_reference_time:
                coordinates.weldx.time_ref = self.reference_time
            return coordinates
        if "time" not in self.coordinates.dims:  # don't interpolate static
            return self.coordinates
        if time.max() <= self.time.min():  # only use edge timestamp
            return self.coordinates.isel(time=0).data
        if time.min() >= self.time.max():  # only use edge timestamp
            return self.coordinates.isel(time=-1).data
        # full interpolation with overlapping times
        return ut.xr_interp_coordinates_in_time(self.coordinates, time)

    def interp_time(
        self,
        time: types_time_like,
        time_ref: types_timestamp_like = None,
    ) -> LocalCoordinateSystem:
        """Interpolates the data in time.

        Note that the returned system won't be time dependent anymore if only a single
        time value was passed. The resulting system is constant and the passed time
        value will be stripped from the result.
        Additionally, if the passed time range does not overlap with the time range of
        the coordinate system, the resulting system won't be time dependent neither
        because the values outside the coordinate systems time range are considered
        as being constant.

        Parameters
        ----------
        time :
            Target time values. If `None` is passed, no interpolation will be performed.
        time_ref:
            The reference timestamp

        Returns
        -------
        LocalCoordinateSystem
            Coordinate system with interpolated data

        """
        if (not self.is_time_dependent) or (time is None):
            return self

        time = Time(time, time_ref)

        if self.has_reference_time != time.is_absolute:
            raise TypeError(
                "Only 1 reference time provided for time dependent coordinate "
                "system. Either the reference time of the coordinate system or the "
                "one passed to the function is 'None'. Only cases where the "
                "reference times are both 'None' or both contain a timestamp are "
                "allowed. Also check that the reference time has the correct type."
            )

        orientation = self._interp_time_orientation(time)
        coordinates = self._interp_time_coordinates(time)

        # remove time if orientations and coordinates are single values (static)
        if orientation.ndim == 2 and coordinates.ndim == 1:
            time = None

        return LocalCoordinateSystem(orientation, coordinates, time)

    def invert(self) -> LocalCoordinateSystem:
        """Get a local coordinate system defining the parent in the child system.

        Inverse is defined as orientation_new=orientation.T,
        coordinates_new=orientation.T*(-coordinates)

        Returns
        -------
        LocalCoordinateSystem
            Inverted coordinate system.

        """
        if isinstance(self.coordinates, TimeSeries):
            raise WeldxException(
                "Can not invert coordinates that are described by an expression. "
                "Use 'interp_time' to create discrete values."
            )
        orientation = ut.xr_transpose_matrix_data(self.orientation, dim1="c", dim2="v")
        coordinates = ut.xr_matmul(
            self.orientation,
            self.coordinates * -1,
            dims_a=["c", "v"],
            dims_b=["c"],
            trans_a=True,
        )
        return LocalCoordinateSystem(
            orientation, coordinates, self.time, self.reference_time
        )

    def plot(
        self,
        axes: matplotlib.axes.Axes = None,
        color: str = None,
        label: str = None,
        time: types_time_like = None,
        time_ref: types_timestamp_like = None,
        time_index: int = None,
        scale_vectors: float | list | np.ndarray = None,
        show_origin: bool = True,
        show_trace: bool = True,
        show_vectors: bool = True,
    ):
        """Plot the coordinate system.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The target matplotlib axes object that should be drawn to. If `None` is
            provided, a new one will be created.
        color : str
            The color of the coordinate system. The string must be a valid matplotlib
            color format. See:
            https://matplotlib.org/3.1.0/api/colors_api.html#module-matplotlib.colors
        label : str
            The name of the coordinate system
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The time steps that should be plotted. Missing time steps in the data will
            be interpolated.
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`
        time_index: int
            If the coordinate system is time dependent, this parameter can be used
            to select a specific key frame by its index.
        scale_vectors :
            A scaling factor or array to adjust the vector length
        show_origin: bool
            If `True`, a small dot with the assigned color will mark the coordinate
            systems' origin.
        show_trace : bool
            If `True`, the trace of time dependent coordinate systems is plotted.
        show_vectors : bool
            If `True`, the coordinate cross of time dependent coordinate systems is
            plotted.

        """
        from weldx.visualization import plot_local_coordinate_system_matplotlib

        plot_local_coordinate_system_matplotlib(
            self,
            axes=axes,
            color=color,
            label=label,
            time=time,
            time_ref=time_ref,
            time_index=time_index,
            scale_vectors=scale_vectors,
            show_origin=show_origin,
            show_trace=show_trace,
            show_vectors=show_vectors,
        )

    def reset_reference_time(self, time_ref_new: pd.Timestamp):
        """Reset the reference time of the coordinate system.

        The time values of the coordinate system are adjusted to the new reference time.
        If no reference time has been set before, the time values will remain
        unmodified. This assumes that the current time delta values are already
        referring to the new reference time.

        Parameters
        ----------
        time_ref_new: pandas.Timestamp
            The new reference time

        """
        self._time_ref = time_ref_new
        self._dataset.weldx.time_ref = time_ref_new
