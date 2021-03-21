"""Contains methods and classes for coordinate transformations."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation as Rot

import weldx.util as ut

from .types import (
    types_coordinates,
    types_orientation,
    types_time_and_lcs,
    types_timeindex,
)
from .util import build_time_index, normalize

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes

__all__ = ("LocalCoordinateSystem",)


class LocalCoordinateSystem:
    """Defines a local cartesian coordinate system in 3d.

    Notes
    -----
    Learn how to use this class by reading the
    :doc:`Tutorial <../tutorials/transformations_01_coordinate_systems>`.

    """

    def __init__(
        self,
        orientation: types_orientation = None,
        coordinates: types_coordinates = None,
        time: types_timeindex = None,
        time_ref: pd.Timestamp = None,
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
            Coordinates of the origin
        time :
            Time data for time dependent coordinate systems. If the provided coordinates
            and orientations contain only a single value, the coordinate system is
            considered to be static and the provided value won't be stored. If this
            happens, a warning will be emitted.
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.
        construction_checks :
            If 'True', the validity of the data will be verified

        Returns
        -------
        LocalCoordinateSystem
            Cartesian coordinate system

        """
        if orientation is None:
            orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if coordinates is None:
            coordinates = np.array([0, 0, 0])

        time, time_ref = build_time_index(time, time_ref)
        orientation = self._build_orientation(orientation, time)
        coordinates = self._build_coordinates(coordinates, time)

        if time is not None and not (
            "time" in coordinates.coords or "time" in orientation.coords
        ):
            warnings.warn(
                "Neither the coordinates nor the orientation are time dependent. "
                "Provided time is dropped"
            )

        if construction_checks:
            ut.xr_check_coords(
                coordinates,
                dict(
                    c={"values": ["x", "y", "z"]},
                    time={"dtype": "timedelta64", "optional": True},
                ),
            )

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
            )

            # vectorize test if orthogonal
            if not ut.xr_is_orthogonal_matrix(orientation, dims=["c", "v"]):
                raise ValueError("Orientation vectors must be orthogonal")

        # unify time axis
        if (
            ("time" in orientation.coords)
            and ("time" in coordinates.coords)
            and (not np.all(orientation.time.data == coordinates.time.data))
        ):
            time_union = ut.get_time_union([orientation, coordinates])
            orientation = ut.xr_interp_orientation_in_time(orientation, time_union)
            coordinates = ut.xr_interp_coordinates_in_time(coordinates, time_union)

        coordinates.name = "coordinates"
        orientation.name = "orientation"

        self._dataset = xr.merge([coordinates, orientation], join="exact")
        if "time" in self._dataset and time_ref is not None:
            self._dataset.weldx.time_ref = time_ref

    def __repr__(self):
        """Give __repr_ output in xarray format."""
        return self._dataset.__repr__().replace(
            "<xarray.Dataset", "<LocalCoordinateSystem"
        )

    def __add__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
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

        rhs_cs = rhs_cs.interp_time(lhs_cs.time, time_ref)

        orientation = ut.xr_matmul(
            rhs_cs.orientation, lhs_cs.orientation, dims_a=["c", "v"]
        )
        coordinates = (
            ut.xr_matmul(rhs_cs.orientation, lhs_cs.coordinates, ["c", "v"], ["c"])
            + rhs_cs.coordinates
        )
        return LocalCoordinateSystem(orientation, coordinates, time_ref=time_ref)

    def __sub__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
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

    def __eq__(self: "LocalCoordinateSystem", other: "LocalCoordinateSystem") -> bool:
        """Check equality of LocalCoordinateSystems."""
        return (
            self.orientation.identical(other.orientation)
            and self.coordinates.identical(other.coordinates)
            and self.reference_time == other.reference_time
        )

    @staticmethod
    def _build_orientation(
        orientation: types_orientation,
        time: pd.DatetimeIndex = None,
    ):
        """Create xarray orientation from different formats and time-inputs.

        Parameters
        ----------
        orientation :
            Orientation object or data.
        time :
            Valid time index formatted with `_build_time_index`.

        Returns
        -------
        xarray.DataArray

        """
        if not isinstance(orientation, xr.DataArray):
            time_orientation = None
            if isinstance(orientation, Rot):
                orientation = orientation.as_matrix()
            elif not isinstance(orientation, np.ndarray):
                orientation = np.array(orientation)

            if orientation.ndim == 3:
                time_orientation = time
            orientation = ut.xr_3d_matrix(orientation, time_orientation)

        # make sure we have correct "time" format
        orientation = orientation.weldx.time_ref_restore()

        return orientation

    @staticmethod
    def _build_coordinates(coordinates, time: pd.DatetimeIndex = None):
        """Create xarray coordinates from different formats and time-inputs.

        Parameters
        ----------
        coordinates:
            Coordinates data.
        time:
            Valid time index formatted with `_build_time_index`.

        Returns
        -------
        xarray.DataArray

        """
        if not isinstance(coordinates, xr.DataArray):
            time_coordinates = None
            if not isinstance(coordinates, (np.ndarray, pint.Quantity)):
                coordinates = np.array(coordinates)
            if coordinates.ndim == 2:
                time_coordinates = time
            coordinates = ut.xr_3d_vector(coordinates, time_coordinates)

        # make sure we have correct "time" format
        coordinates = coordinates.weldx.time_ref_restore()

        return coordinates

    @classmethod
    def from_euler(
        cls, sequence, angles, degrees=False, coordinates=None, time=None, time_ref=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from an euler sequence.

        This function uses scipy.spatial.transform.Rotation.from_euler method to define
        the coordinate systems orientation. Take a look at it's documentation, if some
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
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        orientation = Rot.from_euler(sequence, angles, degrees)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_orientation(
        cls, orientation, coordinates=None, time=None, time_ref=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from orientation matrix.

        Parameters
        ----------
        orientation :
            Orthogonal transformation matrix
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_xyz(
        cls, vec_x, vec_y, vec_z, coordinates=None, time=None, time_ref=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from 3 vectors defining the orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_x = ut.to_float_array(vec_x)
        vec_y = ut.to_float_array(vec_y)
        vec_z = ut.to_float_array(vec_z)

        orientation = np.concatenate((vec_x, vec_y, vec_z), axis=vec_x.ndim - 1)
        orientation = np.reshape(orientation, (*vec_x.shape, 3))
        orientation = orientation.swapaxes(orientation.ndim - 1, orientation.ndim - 2)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_xy_and_orientation(
        cls,
        vec_x,
        vec_y,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_z = cls._calculate_orthogonal_axis(vec_x, vec_y) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

    @classmethod
    def from_yz_and_orientation(
        cls,
        vec_y,
        vec_z,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_x = cls._calculate_orthogonal_axis(vec_y, vec_z) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

    @classmethod
    def from_xz_and_orientation(
        cls,
        vec_x,
        vec_z,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_y = cls._calculate_orthogonal_axis(vec_z, vec_x) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

    @staticmethod
    def _sign_orientation(positive_orientation):
        """Get -1 or 1 depending on the coordinate systems orientation.

        Parameters
        ----------
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not

        Returns
        -------
        int
            1 if the coordinate system has positive orientation,
            -1 otherwise

        """
        if positive_orientation:
            return 1
        return -1

    @staticmethod
    def _calculate_orthogonal_axis(a_0, a_1):
        """Calculate an axis which is orthogonal to two other axes.

        The calculated axis has a positive orientation towards the other 2
        axes.

        Parameters
        ----------
        a_0 :
            First axis
        a_1 :
            Second axis

        Returns
        -------
        numpy.ndarray
            Orthogonal axis

        """
        return np.cross(a_0, a_1)

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
    def coordinates(self) -> xr.DataArray:
        """Get the coordinate systems coordinates.

        Returns
        -------
        xarray.DataArray
            Coordinates of the coordinate system

        """
        return self.dataset.coordinates

    @property
    def is_time_dependent(self) -> bool:
        """Return `True` if the coordinate system is time dependent.

        Returns
        -------
        bool :
            `True` if the coordinate system is time dependent, `False` otherwise.

        """
        return self.time is not None

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
    def reference_time(self) -> Union[pd.Timestamp, None]:
        """Get the coordinate systems reference time.

        Returns
        -------
        pandas.Timestamp:
            The coordinate systems reference time

        """
        return self._dataset.weldx.time_ref

    @property
    def datetimeindex(self) -> Union[pd.DatetimeIndex, None]:
        """Get the time as 'pandas.DatetimeIndex'.

        If the coordinate system has no reference time, 'None' is returned.

        Returns
        -------
        Union[pandas.DatetimeIndex, None]:
            The coordinate systems time as 'pandas.DatetimeIndex'

        """
        if not self.has_reference_time:
            return None
        return self.time + self.reference_time

    @property
    def time(self) -> Union[pd.TimedeltaIndex, None]:
        """Get the time union of the local coordinate system (None if system is static).

        Returns
        -------
        pandas.TimedeltaIndex
            DateTimeIndex-like time union

        """
        if "time" in self._dataset.coords:
            return self._dataset.time
        return None

    @property
    def time_quantity(self) -> pint.Quantity:
        """Get the time as 'pint.Quantity'.

        Returns
        -------
        pint.Quantity:
            The coordinate systems time as 'pint.Quantity'

        """
        return ut.pandas_time_delta_to_quantity(self.time)

    @property
    def dataset(self) -> xr.Dataset:
        """Get the underlying xarray.Dataset with ordered dimensions.

        Returns
        -------
        xarray.Dataset
            xarray Dataset with coordinates and orientation as DataVariables.

        """
        return self._dataset.transpose(..., "c", "v")

    @property
    def is_unity_translation(self) -> bool:
        """Return true if the LCS has a zero translation/coordinates value."""
        if self.coordinates.shape[-1] == 3 and np.allclose(
            self.coordinates, np.zeros(3)
        ):
            return True
        return False

    @property
    def is_unity_rotation(self) -> bool:
        """Return true if the LCS represents a unity rotation/orientations value."""
        if self.orientation.shape[-2:] == (3, 3) and np.allclose(
            self.orientation, np.eye(3)
        ):
            return True
        return False

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

    def interp_time(
        self,
        time: types_time_and_lcs,
        time_ref: Union[pd.Timestamp, None] = None,
    ) -> "LocalCoordinateSystem":
        """Interpolates the data in time.

        Parameters
        ----------
        time :
            Series of times.
            If passing "None" no interpolation will be performed.
        time_ref:
            The reference timestamp

        Returns
        -------
        LocalCoordinateSystem
            Coordinate system with interpolated data

        """
        if (not self.is_time_dependent) or (time is None):
            return self

        # use LCS reference time if none provided
        if isinstance(time, LocalCoordinateSystem) and time_ref is None:
            time_ref = time.reference_time
        time = ut.to_pandas_time_index(time)

        if self.has_reference_time != (
            time_ref is not None or isinstance(time, pd.DatetimeIndex)
        ):
            raise TypeError(
                "Only 1 reference time provided for time dependent coordinate "
                "system. Either the reference time of the coordinate system or the "
                "one passed to the function is 'None'. Only cases where the "
                "reference times are both 'None' or both contain a timestamp are "
                "allowed. Also check that the reference time has the correct type."
            )

        if self.has_reference_time and (not isinstance(time, pd.DatetimeIndex)):
            time = time + time_ref

        orientation = ut.xr_interp_orientation_in_time(self.orientation, time)
        coordinates = ut.xr_interp_coordinates_in_time(self.coordinates, time)

        return LocalCoordinateSystem(orientation, coordinates, time_ref=time_ref)

    def invert(self) -> "LocalCoordinateSystem":
        """Get a local coordinate system defining the parent in the child system.

        Inverse is defined as orientation_new=orientation.T,
        coordinates_new=orientation.T*(-coordinates)

        Returns
        -------
        LocalCoordinateSystem
            Inverted coordinate system.

        """
        orientation = ut.xr_transpose_matrix_data(self.orientation, dim1="c", dim2="v")
        coordinates = ut.xr_matmul(
            self.orientation,
            -self.coordinates,
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
        time: types_time_and_lcs = None,
        time_ref: pd.Timestamp = None,
        time_index: int = None,
        scale_vectors: Union[float, List, np.ndarray] = None,
        show_origin: bool = True,
        show_trace: bool = True,
        show_vectors: bool = True,
    ):  # pragma: no cover
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
            If the coordinate system is time dependent, this parameter can be used to
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
        self._dataset.weldx.time_ref = time_ref_new
